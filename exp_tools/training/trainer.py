import copy
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.amp import autocast, GradScaler

from exp_tools.basic_utils import predict, separate


class History:
    """Represent the history of the training process."""

    def __init__(self, loggers=None):
        self._history = {"train": {"loss": []}, "val": {"loss": []}}
        self.loggers = loggers
        self.__finished = False

    def add_metric(self, name, metric, train=True):
        """Adds a new training/validation metric for an epoch."""
        if self.__finished:
            raise AttributeError("The training is finished, cannot add more data.")
        section = "train" if train else "val"
        current_values = self._history[section].setdefault(name, [])
        current_values.append(metric)
        if self.loggers is not None:
            metric_logs = dict()
            if not isinstance(metric, dict):
                metric_logs[f"{section}/{name}"] = metric
            else:
                for label, value in metric.items():
                    metric_logs[f"{section}_{name}/{label}"] = value
            for logger in self.loggers:
                logger.log_metric(metric_logs, epoch=len(current_values))

    def get_metric(self, name, epoch=-1, train=True):
        """Returns the value of a particular metric at a certain epoch."""
        prefix = "train" if train else "val"
        metric_value = self._history[prefix][name]
        if isinstance(metric_value, list):
            return metric_value[epoch]
        else:
            return {key: value[epoch] for key, value in metric_value.items()}

    @staticmethod
    def _vectorize_list_of_dict(list_of_dicts):
        """Converts and returns a list of dicts to a dict of lists."""
        keys = list_of_dicts[0].keys()
        final_dict = {
            key: [value_dict[key] for value_dict in list_of_dicts] for key in keys
        }
        return final_dict

    def _vectorize(self):
        """Vectorizes all the metrics if they are list of dicts."""
        for metric in self._history["train"]:
            value = self._history["train"][metric]
            if len(value) > 0 and isinstance(value[0], dict):
                self._history["train"][metric] = History._vectorize_list_of_dict(value)
                val_metrics = self._history["val"].get(metric, [])
                if len(val_metrics) > 0:
                    self._history["val"][metric] = History._vectorize_list_of_dict(
                        val_metrics
                    )

    def finish(self):
        """Finalizes the history and freezes the state."""
        self._vectorize()
        self.__finished = True

    @property
    def history(self):
        """Returns the training history as a python dict."""
        return copy.deepcopy(self._history)

    @property
    def epochs(self):
        """Returns the number of epochs."""
        return len(self._history["train"]["loss"])

    def _generate_multiplot(self, axis, values, train=True, metric_title=None):
        """Generates a single plot for the training history metric (2D)."""
        x = range(self.epochs)
        line_style = "-" if train else "--"
        prefix = "train" if train else "val"
        axis.set_xlabel("Epochs")
        axis.set_ylabel(metric_title.capitalize())
        cmap = plt.cm.get_cmap("tab10", len(values))
        for i, (key, value) in enumerate(values.items()):
            label = f"{prefix}_{metric_title}:{key}"
            axis.plot(x, value, label=label, color=cmap(i), linestyle=line_style)
        axis.legend(loc="best")
        axis.set_title(f"{prefix.capitalize()} {metric_title.capitalize()}")

    def _generate_single_plot(self, axis, metric_values, metric_title=None):
        """Generates a single plot for the training history metric (1D)."""
        x = range(self.epochs)
        train_metric = metric_values["train"]
        val_metric = metric_values.get("val", [])
        axis.set_xlabel("Epochs")
        axis.set_ylabel(metric_title.capitalize())
        axis.plot(x, train_metric, label=f"train_{metric_title}", linestyle="-")
        if len(val_metric) > 0:
            axis.plot(x, val_metric, label=f"val_{metric_title}", linestyle="--")
            axis.legend(loc="best")
        axis.set_title(f"{metric_title.capitalize()}")

    @classmethod
    def from_training_dict(cls, training_dict):
        history = cls()
        history._history = training_dict
        history.finish()
        return history

    def plot_history(self):
        """Plots the complete training history."""
        num_plots = len(self._history["train"])
        num_vectored_metrics = sum(
            isinstance(values, dict) for values in self._history["val"].values()
        )
        num_plots += num_vectored_metrics
        num_rows = math.ceil(num_plots / 2)
        fig, axes = plt.subplots(num_rows, 2, figsize=(12, num_rows * 5))
        if not isinstance(axes, np.ndarray):
            axes = np.array([axes])

        axes = axes.flatten()
        i = 0
        for metric, train_metric in self._history["train"].items():
            val_metric = self._history["val"].get(metric, dict())
            if isinstance(train_metric, dict):
                self._generate_multiplot(axes[i], train_metric, metric_title=metric)
                if len(val_metric) > 0:
                    i += 1
                    self._generate_multiplot(
                        axes[i], val_metric, False, metric_title=metric
                    )
            else:
                metrics = {"train": train_metric, "val": val_metric}
                self._generate_single_plot(axes[i], metrics, metric_title=metric)
            i += 1


class Trainer:
    """A simple trainer class to train the model."""

    def __init__(
        self,
        max_epochs=10,
        clip_grad=False,
        clip_val=1.0,
        device="cpu",
        metrics=None,
        history=None,
        checkpoint_interval=5,
        checkpoint_path="./checkpoints/",
        mixed_precision=False,
    ):
        self._max_epochs = max_epochs
        self._device = torch.device(device)
        self.clip_grad = clip_grad
        if not isinstance(history, History):
            self.current_history = History()
        self.metrics = metrics
        self.clip_val = clip_val
        self._checkpoint_interval = checkpoint_interval
        os.makedirs(checkpoint_path, exist_ok=True)
        self._checkpoint_path = checkpoint_path
        self._mixed_precision = mixed_precision

    def fit(self, model, loss_fn, optimizer, train_data, val_data=None):
        """Fits the model to the data."""
        self.reset_history()
        model = model.to(self._device)
        scaler = None
        if self._mixed_precision:
            scaler = GradScaler()
        for epoch in range(1, self._max_epochs + 1):
            print(f"Epoch {epoch + 1}/{self._max_epochs}")
            self._fit_epoch(model, loss_fn, optimizer, train_data, scaler)
            print(f"\nTrain loss: {self.current_history.get_metric('loss')}")
            if val_data:
                self._validate(model, loss_fn, val_data)
                print(
                    f"Val loss: {self.current_history.get_metric('loss', train=False)}"
                )
            # Checkpoint support
            if epoch % self._checkpoint_interval == 0 or epoch == self._max_epochs:
                state = model.state_dict()
                torch.save(state, f"{self._checkpoint_path}/model_{epoch}.pth")
        # Model logging support
        for logger in self.current_history.loggers:
            if logger.allow_model_logging:
                logger.log_model(f"{self._checkpoint_path}/model_{epoch}.pth")
        self.current_history.finish()

    def _fit_epoch(self, model, loss_fn, optimizer, train_data, scaler=None):
        """Fits one single epoch."""
        num_batches = len(train_data)
        loss_vals = []
        n_samples_processed = 0
        # Fitting the model for an epoch
        for batch, (X, y) in enumerate(train_data):
            optimizer.zero_grad()
            X, y = X.to(device=self._device), y.to(self._device)
            if self._mixed_precision:
                with autocast(device_type="cuda"):
                    y_hat = model(X)
                    loss = loss_fn(y_hat, y)
                scaler.scale(loss).backward()
            else:
                y_hat = model(X)
                loss = loss_fn(y_hat, y)
                loss.backward()
            if self.clip_grad:
                if self._mixed_precision:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), max_norm=self.clip_val
                )

            if self._mixed_precision:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            loss_val = loss.item() * y.size(0)
            n_samples_processed += y.size(0)
            loss_vals.append(loss_val)
            Trainer.show_progress(batch + 1, num_batches)

        # Recording the loss
        self.current_history.add_metric("loss", sum(loss_vals) / n_samples_processed)
        # Recording the metrics
        self._record_metrics(model, train_data)

    def _validate(self, model, loss_fn, val_data):
        """Validates the model on validation data."""
        sample_count = 0
        batch_losses = []
        # Evaluating the loss
        with torch.no_grad():
            for X, y in val_data:
                X, y = X.to(device=self._device), y.to(self._device)
                y_hat = model(X)
                loss = loss_fn(y_hat, y).item()
                loss = loss * y.size(0)
                sample_count += y.size(0)
                batch_losses.append(loss)
        # Recording the loss
        self.current_history.add_metric("loss", sum(batch_losses) / sample_count, False)
        # Recording the metrics
        self._record_metrics(model, val_data, False)

    def _record_metrics(self, model, data, train=True):
        """Records the metrics for the model on the data, if user provided."""
        if not self.metrics:
            return
        # Accumulating the predictions
        y_true_total = []
        y_pred_total = []
        with torch.no_grad():
            for X, y in data:
                X, y = X.to(device=self._device), y.to(device=self._device)
                y_preds = predict(model, X)
                y_true_total.append(separate(y))
                y_pred_total.append(separate(y_preds))
        y_true_total = np.concatenate(y_true_total)
        y_pred_total = np.concatenate(y_pred_total)
        # Recording the metrics
        for metric, fun in self.metrics.items():
            self.current_history.add_metric(
                metric, fun(y_true_total, y_pred_total), train
            )

    def reset_history(self):
        """Resets the current history dictionary."""
        self.current_history = History()

    @staticmethod
    def show_progress(current_batch, total_batches):
        n_dashes = current_batch * 50 // total_batches
        percent_complete = current_batch * 100 / total_batches
        n_dots = 50 - n_dashes
        print(
            f"\r[{'-' * n_dashes}{'.' * n_dots}] - batch: {current_batch}/{total_batches} - {percent_complete:.2f} complete",
            end="",
        )
