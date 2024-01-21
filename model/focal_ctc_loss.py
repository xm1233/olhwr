import torch
import torch.nn as nn


class focal_loss(nn.Module):
    def __init__(self):
        super(focal_loss, self).__init__()
        self.ctc = nn.CTCLoss()

    def forward(self, out, text, pre_len, lengths):
        loss = self.ctc(out, text, pre_len, lengths)
        weight = torch.exp(-loss)
        weight = torch.subtract(torch.Tensor([1.0]).cuda(), weight)
        weight = torch.square(weight)
        loss = torch.multiply(weight, loss)
        loss = loss.mean()
        return loss
