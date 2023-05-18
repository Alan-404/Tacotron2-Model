import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor

from model.utils.net import PreNet
from model.utils.attention import LocationSensitiveAttention

from torch.autograd import Variable

class Decoder(nn.Module):
    def __init__(self,
                n_mel_channels: int,
                attention_rnn_dim: int,
                attention_dim: int,
                embedding_dim: int,
                prenet_dim: int,
                decoder_rnn_dim: int,
                n_filters: int,
                kernel_size: int,
                p_attention_dropout: float,
                p_decoder_dropout: float) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_mel_channels = n_mel_channels
        self.attention_rnn_dim = attention_rnn_dim
        self.decoder_rnn_dim = decoder_rnn_dim
        self.prenet_dim = prenet_dim
        self.p_attention_dropout = p_attention_dropout
        self.p_decoder_dropout = p_decoder_dropout

        self.pre_net = PreNet(
            in_dim=n_mel_channels,
            sizes=[prenet_dim, prenet_dim]
        )

        self.attention_rnn = nn.LSTMCell(
            input_size=prenet_dim + embedding_dim,
            hidden_size=attention_rnn_dim
        )

        self.attention = LocationSensitiveAttention(
            attention_rnn_dim=attention_rnn_dim,
            embedding_dim=embedding_dim,
            attention_dim=attention_dim,
            n_filters=n_filters,
            kernel_size=kernel_size
        )

        self.decoder_rnn = nn.LSTMCell(
            input_size=attention_rnn_dim + embedding_dim,
            hidden_size=decoder_rnn_dim
        )

        self.linear_projection = nn.Linear(in_features=decoder_rnn_dim+embedding_dim, out_features=n_mel_channels)
        self.gate_layer = nn.Linear(in_features=decoder_rnn_dim + embedding_dim, out_features=1)


    def init_frame(self, memory: Tensor):
        decoder_input = Variable(memory.data.new(memory.size(0), self.n_mel_channels).zero_())

        return decoder_input
    
    def init_states(self, memory: Tensor, mask: Tensor):
        batch_size = memory.size(0)
        n_ctx = memory.size(1)

        # attention rnn
        self.attention_hidden = Variable(memory.data.new(batch_size, self.attention_rnn_dim).zero_())
        self.attention_cell = Variable(memory.data.new(batch_size, self.attention_rnn_dim).zero_())

        # decoder rnn
        self.decoder_hidden = Variable(memory.data.new(batch_size, self.decoder_rnn_dim).zero_())
        self.decoder_cell = Variable(memory.data.new(batch_size, self.decoder_rnn_dim).zero_())

        # attention weights
        self.attention_weights = Variable(memory.data.new(batch_size, n_ctx).zero_())
        self.attention_weights_cumulative = Variable(memory.data.new(batch_size, n_ctx).zero_())
        self.attention_context = Variable(memory.data.new(batch_size, self.embedding_dim).zero_())

        # encoder handle
        self.memory = memory
        self.mask = mask



    def decode(self, decoder_input: Tensor):
        """ 
            decoder_input: Tensor
            - (batch_size, prenet_dim) 
        """

        # attention rnn
        attention_input = torch.cat((decoder_input, self.attention_context), dim=-1) # (batch_size, prenet_dim + embedding_dim)
        self.attention_hidden, self.attention_cell = self.attention_rnn(attention_input, (self.attention_hidden, self.attention_cell)) # (batch_size, attention_rnn_dim)
        self.attention_hidden = F.dropout(self.attention_hidden, p=self.p_attention_dropout, training=self.training)

        # attention location
        attention_weights_cat = torch.cat((self.attention_weights.unsqueeze(1), self.attention_weights_cumulative.unsqueeze(1)), dim=1) # (batch_size, 2, n_ctx)
        self.attention_context, self.attention_weights = self.attention(self.attention_hidden, self.memory, attention_weights_cat, self.mask) # (batch_size, embedding_dim) (batch_size, n_ctx)
        self.attention_weights_cumulative += self.attention_weights # (batch_size, n_ctx)

        # decoder rnn
        decoder_input = torch.cat((self.attention_hidden, self.attention_context), dim=-1) # (batch_size, attention_rnn_dim + embedding_dim)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(decoder_input, (self.decoder_hidden, self.decoder_cell)) # (batch_size, decoder_rnn_dim)
        self.decoder_hidden = F.dropout(self.decoder_hidden, p=self.p_decoder_dropout, training=self.training)

        # linear projection
        decoder_hidden_context = torch.cat((self.decoder_hidden, self.attention_context), dim=-1) # (batch_size, decoder_rnn_dim + embedding_dim)
        decoder_output = self.linear_projection(decoder_hidden_context) # (batch_size, n_mel_channels)
        gate_output = self.gate_layer(decoder_hidden_context) # (batch_size, 1)
        return decoder_output, gate_output # (batch_size, n_mel_channels) (batch_size, 1) (batch_size, n_ctx)

    def parse_decoder_outputs(self, mel_outputs: list, gate_outputs: list):
        mel_outputs = torch.stack(mel_outputs).contiguous() # (time_out, batch_size, n_mel_channels)
        mel_outputs = mel_outputs.permute(1, 2, 0)

        gate_outputs = torch.stack(gate_outputs).squeeze(-1).contiguous() # (time_out, batch_size)
        gate_outputs = gate_outputs.transpose(0,1)

        return mel_outputs, gate_outputs

    
    def forward(self, memory: Tensor, decoder_inputs: Tensor, mask: Tensor):
        """ 
            memory: Tensor
            - (batch_size, n_ctx, embedding_dim)
            decoder_inputs: Tensor
            - (batch_size, n_mel_channels, time_out) 
        """
        decoder_inputs = decoder_inputs.permute(2, 0, 1) # (time_out, batch_size, n_mel_channels)
        decoder_input = self.init_frame(memory) # (batch_size, n_mel_channels)
        decoder_inputs = torch.cat((decoder_input.unsqueeze(0), decoder_inputs), dim=0) # (time_out + 1, batch_size, n_mel_channels)
        
        
        decoder_inputs = self.pre_net(decoder_inputs) # (time_out + 1, batch_size, prenet_dim)
        
        
        self.init_states(memory, mask)

        mel_outputs, gate_outputs = [], []
        
        for index in range(decoder_inputs.size(0) - 1):
            decoder_input = decoder_inputs[index] # (batch_size, n_mel_channels)

            mel_output, gate_output = self.decode(decoder_input)
            
            mel_outputs += [mel_output]
            gate_outputs += [gate_output]
        """ 
            After loop:
            - mel_outputs: list has shape: (time_out, batch_size, n_mel_channels)
            - gate_outputs: list has shape: (time_out, batch_size, 1)
        """
        
        mel_outputs, gate_outputs = self.parse_decoder_outputs(mel_outputs, gate_outputs)
    
        return mel_outputs, gate_outputs
    
    def inference(self, memory: Tensor, max_decoder_steps: int, gate_threshold: float):
        decoder_input = self.init_frame(memory) # (batch_size, n_mel_channels)

        self.init_states(memory, mask=None)

        mel_outputs, gate_outputs = [], []

        for _ in range(max_decoder_steps):
            decoder_input = self.pre_net(decoder_input)
            mel_output, gate_output = self.decode(decoder_input)

            mel_outputs += [mel_output.squeeze(1)]
            gate_outputs += [gate_output]

            if torch.sigmoid(gate_output.data) < gate_threshold:
                break
            
            decoder_input = mel_output

        mel_outputs, gate_outputs = self.parse_decoder_outputs(mel_outputs, gate_outputs)

        return mel_outputs, gate_outputs
