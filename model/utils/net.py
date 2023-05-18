import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class PostNet(nn.Module):
    def __init__(self, n_convolutions: int, n_mel_channels: int, embedding_dim: int, kernel_size: int) -> None:
        super().__init__()
        self.n_convolutions = n_convolutions
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(in_channels=n_mel_channels, out_channels=embedding_dim, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2), dilation=1),
                nn.BatchNorm1d(num_features=embedding_dim)
            )
        )

        for _ in range(1, n_convolutions-1):
            self.convolutions.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2), dilation=1),
                    nn.BatchNorm1d(num_features=embedding_dim)
                )
            )

        self.convolutions.append(
            nn.Sequential(
                nn.Conv1d(in_channels=embedding_dim, out_channels=n_mel_channels, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2), dilation=1),
                nn.BatchNorm1d(num_features=n_mel_channels)
            )
        )

    def forward(self, x: Tensor):
        for index in range(self.n_convolutions-1):
            x = self.convolutions[index](x)
            x = F.tanh(x)
            x = F.dropout(x, p=0.5, training=self.training)
        x = self.convolutions[-1](x)
        x = F.dropout(x, p=0.5, training=self.training)
        return x


class PreNet(nn.Module):
    def __init__(self, in_dim: int, sizes: list) -> None:
        super().__init__()
        in_sizes = [in_dim] + sizes[:-1]

        self.layers = nn.ModuleList([nn.Linear(in_features=in_size, out_features=out_size, bias=False) for (in_size, out_size) in zip(in_sizes, sizes)])

    def forward(self, x: Tensor):
        for layer in self.layers:
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

        return x