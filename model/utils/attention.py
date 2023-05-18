import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

class LocationSensitiveAttention(nn.Module):
    def __init__(self, attention_rnn_dim: int, embedding_dim: int, attention_dim: int, n_filters: int, kernel_size: int) -> None:
        super().__init__()
        self.query_layer = nn.Linear(in_features=attention_rnn_dim, out_features=attention_dim, bias=False)
        self.memory_layer = nn.Linear(in_features=embedding_dim, out_features=attention_dim, bias=False)
        self.v = nn.Linear(in_features=attention_dim, out_features=1)

        self.location = Location(n_filters=n_filters, kernel_size=kernel_size, attention_dim=attention_dim)

    def get_alignment_energies(self, attention_hidden: Tensor, memory: Tensor, attention_weights_cat: Tensor):
        """ 
            attention_hidden: Tensor 
            - (batch_size, attention_rnn_dim)
            memory: Tensor
            - (batch_size, n_ctx, embedding_dim)
            attention_weights_cat: Tensor
            - (batch_size, 2, n_ctx)
        """
        processed_query = self.query_layer(attention_hidden) # (batch_size, attention_dim)
        processed_memory = self.memory_layer(memory) # (batch_size, n_ctx attention_dim)

        processed_attention = self.location(attention_weights_cat) # (batch_size, n_ctx, attention_dim)

        energies = self.v(
            torch.tanh(processed_memory + processed_query.unsqueeze(1) + processed_attention) # (batch_size, n_ctx, attention_dim)
        ) # (batch_size, n_ctx, 1)
        return energies.squeeze(-1) # (batch_size, n_ctx)
    
    def forward(self, attention_hidden: Tensor, memory: Tensor, attention_weights_cat: Tensor, mask: Tensor = None):
        """ 
            attention_hidden: Tensor 
            - (batch_size, attention_rnn_dim)
            memory: Tensor
            - (batch_size, n_ctx, embedding_dim)
            attention_weights_cat: Tensor
            - (batch_size, 2, n_ctx)
            mask: Tensor
            - (batch_size, n_ctx)
        """
        alignments = self.get_alignment_energies(attention_hidden, memory, attention_weights_cat) # (batch_size, n_ctx)

        if mask is not None:
            alignments += mask*(-1e25) # (batch_size, n_ctx)

        attention_weights = F.softmax(alignments) # (batch_size, n_ctx)

        attention_output = torch.matmul(alignments.unsqueeze(1),memory) # (batch_size, 1, embedding_dim)

        return attention_output.squeeze(1), attention_weights.squeeze(1) # (batch_size, embedding_dim) (batch_size, n_ctx)

class Location(nn.Module):
    def __init__(self, n_filters: int, kernel_size: int, attention_dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(in_channels=2, out_channels=n_filters, kernel_size=kernel_size, stride=1, padding=int((kernel_size-1)/2), dilation=1, bias=False)
        self.linear = nn.Linear(in_features=n_filters, out_features=attention_dim, bias=False)

    def forward(self, x: Tensor):
        """ 
            x: Tensor
            - (batch_size, 2, n_ctx)
        """
        x = self.conv(x) # (batch_size, n_filters, n_ctx)
        x = x.transpose(-1,-2) # (batch_size, n_ctx, n_filters)
        x = self.linear(x) # (batch_size, n_ctx, attention_dim)
        return x