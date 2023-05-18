import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from model.tacotron2 import Tacotron2
from model.loss import Loss
from torch import Tensor
from torch.utils.data import TensorDataset, DataLoader
import os
from model.utils.mask import generate_padding_mask

class Tacotron2Trainer:
    def __init__(self,
                token_size: int,
                n_mel_channels: int,
                embedding_dim: int = 512,
                encoder_kernel_size: int = 5,
                attention_dim: int = 128,
                attention_rnn_dim: int = 1024,
                decoder_rnn_dim: int = 1024,
                n_filters: int = 32,
                prenet_dim: int = 256,
                location_kernel_size: int = 5,
                encoder_n_convolutions: int = 3,
                postnet_n_convolutions: int = 5,
                postnet_kernel_size: int = 5,
                p_attention_dropout: float = 0.1,
                p_decoder_dropout: float = 0.1,
                checkpoint: str = None,
                device: str = "cpu") -> None:
        self.model = Tacotron2(
            token_size = token_size,
            n_mel_channels = n_mel_channels,
            embedding_dim = embedding_dim,
            encoder_kernel_size = encoder_kernel_size,
            attention_dim = attention_dim,
            attention_rnn_dim = attention_rnn_dim,
            decoder_rnn_dim = decoder_rnn_dim,
            n_filters = n_filters,
            prenet_dim = prenet_dim,
            location_kernel_size = location_kernel_size,
            encoder_n_convolutions = encoder_n_convolutions,
            postnet_n_convolutions = postnet_n_convolutions,
            postnet_kernel_size = postnet_kernel_size,
            p_attention_dropout = p_attention_dropout,
            p_decoder_dropout = p_decoder_dropout
        )   

        self.epoch = 0

        self.criterion = nn.CrossEntropyLoss()
        self.loss = 0.0
        self.history = []
        self.loss_function = Loss()

        self.optimizer = optim.Adam(params=self.model.parameters())

        self.checkpoint = checkpoint
        self.device = device
        self.model.to(device)
        if self.checkpoint is not None:
            self.load_model(self.checkpoint)

    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": self.epoch,
            'history': self.history
        }, path)

    def load_model(self, path: str):
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.history = checkpoint['history']

    def build_dataset(self, inputs: Tensor, labels: Tensor, batch_size: int, shuffle: bool):
        dataset = TensorDataset(inputs, labels)
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    
    def train_step(self, texts: Tensor, mels: Tensor):
        self.optimizer.zero_grad()

        mel_outputs, postnet_mel_outputs, gate_outputs = self.model(texts, mels)

        mel_mask = ~generate_padding_mask(torch.mean(mels, dim=1), pad_value=0.5)
        loss = self.loss_function(postnet_mel_outputs, mel_outputs, mels, gate_outputs, mel_mask)

        loss.backward()
        self.optimizer.step()
        self.loss += loss.item()

    def fit(self, inputs: Tensor, labels: Tensor, epochs: int = 1, batch_size: int = 1, shuffle: bool = True, mini_batch: int = 1, learning_rate: float = 0.00003):
        for g in self.optimizer.param_groups:
            g['lr'] = learning_rate
        
        dataloader = self.build_dataset(inputs, labels, batch_size=batch_size, shuffle=shuffle)

        total = len(dataloader)
        delta = total-(total//mini_batch)*mini_batch

        epoch_loss = 0.0
        self.model.train()
        for _ in range(epochs):
            for index, data in enumerate(dataloader, 0):
                inputs = data[0].to(self.device)
                labels = data[1].to(self.device)

                self.train_step(inputs, labels)

                if index%mini_batch == mini_batch-1:
                    print(f"Epoch: {self.epoch+1} Batch: {index+1} Loss: {(self.loss/mini_batch):.4f}")
                    epoch_loss += self.loss
                    self.loss = 0.0
                elif index == total-1:
                    print(f"Epoch: {self.epoch+1} Batch: {index+1} Loss: {(self.loss/delta):.4f}")
                    epoch_loss += self.loss
                    self.loss = 0.0

            self.history.append(epoch_loss/total)

            self.epoch += 1

        if self.checkpoint is not None:
            self.save_model(self.checkpoint)

    

    def predict(self, texts: Tensor, max_decoder_steps: int, gate_threshold: float = 0.5):
        self.model.eval()
        mel_outputs, gate_outputs = self.model.inference(texts, max_decoder_steps, gate_threshold)

        return mel_outputs, gate_outputs