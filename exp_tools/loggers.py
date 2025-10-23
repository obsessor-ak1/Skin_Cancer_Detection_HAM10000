from abc import ABC, abstractmethod
import os

import wandb


class Logger(ABC):
    """Abstract base class for loggers."""

    @abstractmethod
    def log_metric(self, metric_values, epoch=-1):
        """Logs metrics for a given epoch, default to last epoch."""

    @abstractmethod
    def log_model(self, model_path, model_name=None):
        """Logs the model parameters."""


class WandBLogger(Logger):
    """A Logger that uses wandb to log metrics."""

    def __init__(
        self, project_name, config, entity, key=None, allow_model_logging=True
    ):
        if not self.is_logged_in:
            wandb.login(key=key)
        self._run = wandb.init(project=project_name, entity=entity, config=config)
        self._is_completed = False
        self._allow_model_logging = allow_model_logging

    @property
    def allow_model_logging(self):
        return self._allow_model_logging

    def log_metric(self, metric_values, epoch=None):
        self._run.log(metric_values, step=epoch)

    def log_model(self, model_path, model_name=None):
        assert self.allow_model_logging
        if model_name is None:
            model_name = self._run.name + "_model"
        self._run.log_model(model_path, name=model_name)

    @property
    def is_logged_in(self):
        return os.environ.get("WANDB_API_KEY") is not None

    def complete(self):
        assert not self._is_completed
        self._run.finish()
        self._is_completed = True
