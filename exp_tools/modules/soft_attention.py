import torch
from torch import nn
import torch.nn.functional as F


class SoftAttentionMap(nn.Module):
    """Defines Soft-Attention Layer for CNN based architectures."""

    def __init__(self, in_channels, attention_maps=16, kernel_size=3):
        super().__init__()
        self.n_attention_maps = attention_maps
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self._map_generator = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.n_attention_maps,
            kernel_size=self.kernel_size,
            padding=kernel_size // 2,
            stride=1,
        )
        self.gamma = nn.Parameter(torch.tensor(0.01))

    def forward(self, X):
        h, w = X.shape[-2:]
        attention_maps = self._map_generator(X)
        attention_maps = attention_maps.view(X.size(0), self.n_attention_maps, -1)
        normalized_map = F.softmax(attention_maps, -1)
        normalized_map = normalized_map.view(X.size(0), self.n_attention_maps, h, w)
        return normalized_map


class SoftAttentionBlock(nn.Module):
    """Represents a Soft-Attention Block's complete implementation."""

    def __init__(
        self,
        in_channels,
        k,
        kernel_size=3,
        gamma=0.01,
        aggregate=True,
        concat=True,
        dropout=0.5,
    ):
        super().__init__()
        self.soft_attention_mapper = SoftAttentionMap(
            in_channels=in_channels,
            attention_maps=k,
            kernel_size=kernel_size,
        )
        self.gamma = nn.Parameter(torch.tensor(gamma))
        self.dropout = nn.Dropout(dropout)
        self.aggregate = aggregate
        self.concat = concat
        self.pooler = nn.MaxPool2d(kernel_size=2)

    def forward(self, X):
        h, w = X.shape[-2:]
        attention_maps = self.soft_attention_mapper(X)
        if self.aggregate:
            aggregated_map = torch.sum(attention_maps, dim=1, keepdim=True)
            final_map = self.gamma * aggregated_map * X
        else:
            attention_maps = attention_maps.unsqueeze(dim=2)
            final_map = attention_maps * X.unsqueeze(dim=1)
            final_map = self.gamma * final_map.reshape(X.size(0), -1, h, w)
        if self.concat:
            final_map = torch.cat((self.pooler(X), self.pooler(final_map)), dim=1)
        return self.dropout(F.relu(final_map))
