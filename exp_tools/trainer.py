import numpy as np
import torch

from exp_tools.basic_utils import predict, separate


class Trainer:
    """A simple trainer class to train the model."""

    def __init__(self, max_epochs=10, clip_grad=False, clip_val=1, device="cuda", metrics=None):
        self._max_epochs = max_epochs
        self._device = torch.device(device)
        self.clip_grad = clip_grad
        self.current_history = {
            "train": dict(),
            "val": dict()
        }
        self.metrics = metrics
        self.clip_val = clip_val

    def fit(self, model, loss_fn, optimizer, train_data, val_data=None):
        """Fits the model to the data."""
        self.reset_history()
        model = model.to(self._device)
        for epoch in range(self._max_epochs):
            print(f"Epoch {epoch + 1}/{self._max_epochs}")
            self._fit_epoch(model, loss_fn, optimizer, train_data)
            print(f"\nTrain loss: {self.current_history['train']['loss'][-1]}")
            if val_data:
                self._validate(model, loss_fn, val_data)
                print(f"Val loss: {self.current_history['val']['loss'][-1]}")

    def _fit_epoch(self, model, loss_fn, optimizer, train_data):
        """Fits one single epoch."""
        num_batches = len(train_data)
        loss_vals = []
        n_samples_processed = 0
        # Fitting the model for an epoch
        for batch, (X, y) in enumerate(train_data):
            X, y = X.to(device=self._device), y.to(self._device)
            y_hat = model(X)
            loss = loss_fn(y_hat, y)
            loss.backward()
            if self.clip_grad:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.clip_val)
            optimizer.step()
            loss_val = loss.item() * y.size(0)
            n_samples_processed += y.size(0)
            loss_vals.append(loss_val)
            Trainer.show_progress(batch + 1, num_batches)

        # Recording the loss
        loss_hist = self.current_history["train"].setdefault("loss", [])
        loss_hist.append(sum(loss_vals) / n_samples_processed)
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
        val_losses = self.current_history["val"].setdefault("loss", [])
        val_losses.append(sum(batch_losses) / sample_count)
        # Recording the metrics
        self._record_metrics(model, val_data, False)

    def _record_metrics(self, model, data, train=True):
        """Records the metrics for the model on the data, if user provided."""
        if not self.metrics:
            return
        history_section = "train" if train else "val"
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
            metric_values = self.current_history[history_section].setdefault(metric, [])
            metric_values.append(fun(y_true_total, y_pred_total))

    def reset_history(self):
        """Resets the current history dictionary."""
        self.current_history["train"].clear()
        self.current_history["val"].clear()

    @staticmethod
    def show_progress(current_batch, total_batches):
        n_dashes = current_batch * 50 // total_batches
        percent_complete = current_batch * 100 / total_batches
        n_dots = 50 - n_dashes
        print(
            f"\r[{'-' * n_dashes}{'.' * n_dots}] - batch: {current_batch}/{total_batches} - {percent_complete:.2f} complete",
            end='')
