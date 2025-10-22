import os
import tempfile

import numpy as np
import torch
from torch import amp
from torch import nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from exp_tools.basic_utils import predict, separate
from exp_tools.data_utils import DistributedWeightedRandomSampler


class DistributedTrainer:
    """A trainer that performs distributed training."""

    def __init__(
        self,
        max_epochs=5,
        batch_size_per_rank=128,
        checkpoint_interval=5,
        mixed_precision=False,
        clip_grad=None,
        metrics=None,
        loggers=None,
    ):
        self._max_epochs = max_epochs
        self._batch_size_per_rank = batch_size_per_rank
        self._checkpoint_interval = checkpoint_interval
        self._clip_grad = clip_grad
        self._metrics = metrics
        self._is_initialized = False
        self._is_ready = False
        self.mixed_precision = mixed_precision
        self._proc_config = dict()
        self._train_config = dict()
        self._loggers = loggers

    def init_config(
        self, host="localhost", port="12355", world_size=None, backend="nccl"
    ):
        """Initialize distributed configuration."""
        os.environ["MASTER_ADDR"] = host
        os.environ["MASTER_PORT"] = port
        if world_size is None:
            world_size = torch.cuda.device_count()
        self._proc_config.update(
            world_size=world_size,
            m_host=host,
            m_port=port,
            backend=backend,
        )
        self._is_initialized = True

    @property
    def is_configured(self):
        return self._is_initialized

    def _setup(self, rank):
        """Sets up a distributed process"""
        assert self.is_configured
        world_size = self._proc_config["world_size"]
        backend = self._proc_config["backend"]
        dist.init_process_group(
            backend=backend,
            rank=rank,
            world_size=world_size,
        )

    def prepare_trainer(
        self, model_class, optimizer_fn, train_set, loss, model_args=None, val_set=None
    ):
        """Prepares the trainer for training"""
        self._train_config["model_class"] = model_class
        self._train_config["optimizer_fn"] = optimizer_fn
        self._train_config["train_set"] = train_set
        self._train_config["val_set"] = val_set
        self._train_config["loss_fn"] = loss
        if model_args is None:
            model_args = dict()
        self._train_config["model_args"] = model_args
        self._is_ready = True

    @property
    def is_ready(self):
        return self._is_ready

    def begin_training(self, rank):
        """A function to begin training session."""
        assert self.is_ready
        # Initializing the process
        self._setup(rank)
        # Initializing the model and broadcasting its parameters
        torch.cuda.set_device(rank)
        model = self._train_config["model_class"](
            **self._train_config["model_args"]
        ).to(rank)
        ddp_model = DDP(model, device_ids=[rank])
        for param in ddp_model.parameters():
            dist.broadcast(param.data, src=0)
        # Initializing the optimizer and loss
        optimizer = self._train_config["optimizer_fn"](ddp_model)
        loss_fn = self._train_config["loss_fn"]
        # Preparing the datasets
        train_set = self._train_config["train_set"]
        val_set = self._train_config["val_set"]
        world_size = self._proc_config["world_size"]
        # Preparing the samplers
        train_label_weights = 1 / train_set.sample_dist
        train_sample_weights = [
            train_label_weights[train_set.label_map[label_str]].item()
            for label_str in train_set.metadata.dx
        ]
        train_sampler = DistributedWeightedRandomSampler(
            weights=train_sample_weights,
            num_samples=len(train_set),
            num_replicas=world_size,
            rank=rank,
        )
        val_sampler = DistributedSampler(
            val_set, num_replicas=world_size, rank=rank, shuffle=False
        )
        # Preparing the loaders
        train_loader = DataLoader(
            train_set, batch_size=self._batch_size_per_rank, sampler=train_sampler
        )
        val_loader = DataLoader(
            val_set, batch_size=self._batch_size_per_rank, sampler=val_sampler
        )
        # Beginning the training loop
        checkpoint_path = None
        if rank == 0:
            checkpoint_path = tempfile.TemporaryDirectory()
        for i in range(1, self._max_epochs + 1):
            if rank == 0:
                print(f"Epoch: {i}/{self._max_epochs}:")
            train_metrics = self._train_epoch(
                ddp_model, optimizer, loss_fn, train_loader, rank
            )
            for metrics in train_metrics:
                dist.all_reduce(train_metrics[metrics], op=dist.ReduceOp.AVG)
            # Printing training status and logging its metrics
            if rank == 0:
                print("Training completed...")
                print(f"Train loss: {train_metrics['train_loss']}")
                for logger in self._loggers:
                    logger.log_metric(train_metrics, epoch=i)
            val_metrics = self._validate(ddp_model, loss_fn, val_loader, rank)
            for metrics in val_metrics:
                dist.all_reduce(val_metrics[metrics], op=dist.ReduceOp.SUM)
            # Printing validation status and logging its metrics
            if rank == 0:
                print("Validation completed...")
                print(f"Val loss: {val_metrics['val_loss']}")
                for logger in self._loggers:
                    logger.log_metric(val_metrics, epoch=i)

            # Checkpointing model
            if rank == 0:
                if i % self._checkpoint_interval == 0 or i == self._max_epochs:
                    state = ddp_model.module.state_dict()
                    ckp_model_name = f"checkpoint_{rank}_{i}.pth"
                    ckp_path = os.path.join(checkpoint_path.name, ckp_model_name)
                    torch.save(state, ckp_path)
                    if i == self._max_epochs:
                        for logger in self._loggers:
                            logger.log_model(ckp_model_name)
                            logger.complete()
                        checkpoint_path.cleanup()
        # Cleaning up the process group
        dist.destroy_process_group()

    def _train_epoch(self, model, optimizer, loss_fn, train_loader, rank):
        """Trains one epoch on a full training set."""
        batch_wise_loss = 0
        num_samples = 0
        model.train()

        def for_pass(model, X, y):
            output = model(X)
            return loss_fn(output, y)

        # Initializing Gradient Scaler for mixed precision training
        scaler = amp.GradScaler()
        # Starting the training loop
        for X, y in train_loader:
            X, y = X.to(rank), y.to(rank)
            optimizer.zero_grad()
            if self.mixed_precision:
                with amp.autocast(device_type="cuda"):
                    loss = for_pass(model, X, y)
                # Using scaled loss (by a factor) during mixed precision for backward pass
                # in case the gradients get too small or large to fit in fp16 type because
                # overflow or underflow during calculation.
                scaler.scale(loss).backward()
            else:
                loss = for_pass(model, X, y)
                loss.backward()

            if self._clip_grad is not None:
                if self.mixed_precision:
                    # Unscaling to get true gradient before clipping
                    scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), self._clip_grad)

            if self.mixed_precision:
                # Optimizing by unscaling the gradients
                scaler.step(optimizer)
                # Updating the scaling factor
                scaler.update()
            else:
                optimizer.step()
            batch_wise_loss += loss.item() * y.size(0)
            num_samples += y.size(0)
        full_train_loss = torch.tensor(batch_wise_loss / num_samples, device=rank)
        # Recording train metrics
        train_metrics = self._record_metrics(model, train_loader, rank, train=True)
        train_metrics["train_loss"] = full_train_loss
        return train_metrics

    def _validate(self, model, loss_fn, val_loader, rank):
        """Validates one epoch on a validation set."""
        batch_wise_loss = 0
        num_samples = 0
        model.eval()
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(rank), y.to(rank)
                loss = loss_fn(model(X), y)
                batch_wise_loss += loss.item() * y.size(0)
                num_samples += y.size(0)

        full_val_loss = torch.tensor(batch_wise_loss / num_samples, device=rank)
        # Recording the validation/test metrics
        val_metrics = self._record_metrics(model, val_loader, rank, train=False)
        val_metrics["val_loss"] = full_val_loss
        return val_metrics

    def _record_metrics(self, model, loader, rank, train=False):
        """Records and returns the configured metrics for the model on the
        loader."""
        metrics = dict()
        y_true_total = []
        y_pred_total = []
        with torch.no_grad():
            for X, y in loader:
                X, y = X.to(rank), y.to(rank)
                y_pred = predict(model, X)
                y_true_total.append(separate(y))
                y_pred_total.append(separate(y_pred))
        y_true_total = np.concatenate(y_true_total)
        y_pred_total = np.concatenate(y_pred_total)
        prefix = "train_" if train else "test_"
        for metric, fun in self._metrics.items():
            metrics[prefix + metric] = torch.tensor(
                fun(y_true_total, y_pred_total), device=rank
            )
        return metrics
