import math

import matplotlib.pyplot as plt
import torch
from torch import nn


def predict(model: nn.Module, data: torch.Tensor):
    """Returns predictions using trained model."""
    # Assuming the model and the data are on the same device
    logits = model(data)
    labels = torch.argmax(logits, dim=1)
    return labels

def separate(tensor: torch.Tensor):
    """Separates the tensor from gradient and returns a numpy array."""
    return tensor.detach().cpu().numpy()


def prepare_model(model: nn.Module, shape, verbose=True):
    """Passes a dummy input to the model to initialize lazy layers
     and visualize output shape."""
    X = torch.randn(shape)
    for name, module in model.named_children():
        X = module(X)
        if verbose:
            print(f"{name}: {type(module).__name__} output shape: {X.shape}")


def init_module(module: nn.Module):
    if isinstance(module, (nn.Linear, nn.Conv2d)):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


def plot_history(history_dict):
    """Plots the training history of the model, including loss and
    any other metric provided."""
    metric_count = len(history_dict["train"])
    n_rows = math.ceil(metric_count / 2)
    plt.figure(figsize=(12, n_rows * 5))
    train_metrics = history_dict["train"]
    val_metrics = history_dict["val"]
    for i, (metric, values) in enumerate(train_metrics.items()):
        plt.subplot(n_rows, 2, i + 1)
        plt.plot(values, label=f"train_{metric}")
        if metric in val_metrics:
            plt.plot(val_metrics[metric], label=f"val_{metric}")
        plt.legend(loc="best")
        plt.title(metric.title())