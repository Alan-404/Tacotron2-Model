import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor


class Encoder(nn.Module):
    def __init__(self, token_size: int, embedding_dim: int, n_convolutions: int, kernel_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings=token_size, embedding_dim=embedding_dim)

        self.convolutions = nn.ModuleList(
            [nn.Sequential(
                nn.Conv1d(in_channels=embedding_dim, out_channels=embedding_dim, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2), dilation=1),
                nn.BatchNorm1d(num_features=embedding_dim)
            ) for _ in range(n_convolutions)]
        )

        self.bidirectional_lstm = nn.LSTM(embedding_dim, int(embedding_dim/2), 1, batch_first=True, bidirectional=True)

    def forward(self, x: Tensor, lengths: Tensor):
        x = self.embedding(x)
        x = x.transpose(-1,-2)

        for layer in self.convolutions:
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(-1,-2)

        lengths = lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            
        self.bidirectional_lstm.flatten_parameters()
        outputs, _ = self.bidirectional_lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=True)

        return outputs
    
    def inference(self, x: Tensor):
        x = self.embedding(x)
        x = x.transpose(-1,-2)

        for layer in self.convolutions:
            x = layer(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        x = x.transpose(-1,-2)

        self.bidirectional_lstm.flatten_parameters()

        outputs, _ = self.bidirectional_lstm(x)

        return outputs