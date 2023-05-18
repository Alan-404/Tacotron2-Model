import torch
from torch import nn
from torch import Tensor


class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()

    def forward(self, postnet_mel_out: Tensor, mel_out: Tensor, mel_target: Tensor, gate_out: Tensor, gate_target: Tensor):
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        
        mel_out = mel_out.masked_fill((gate_target==False).unsqueeze(1), 0.0)
        postnet_mel_out = postnet_mel_out.masked_fill((gate_target==False).unsqueeze(1), 0.0)
        gate = gate_target.type(torch.float32)
        """ gate_out = gate_out.view(-1, 1) """
        loss = nn.MSELoss()(mel_out, mel_target) + nn.MSELoss()(postnet_mel_out, mel_target) + nn.BCEWithLogitsLoss()(gate_out, gate)
    
        return loss