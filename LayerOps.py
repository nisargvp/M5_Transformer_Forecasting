import torch, math
import torch.nn as nn
import torch.nn.functional as F


class FeedForward(nn.Module):
    def __init__(self, input_dim, d_hidden=128, dropout=0.1):
        super().__init__()
        self.lin1 = nn.Linear(input_dim, d_hidden)
        self.dropout = nn.Dropout(dropout)
        self.lin2 = nn.Linear(d_hidden, input_dim)

    def forward(self, x):
        x = self.dropout(F.relu(self.lin1(x)))
        return self.lin2(x)


class Norm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.size = dim
        # create two learnable parameters to calibrate normalisation
        self.alpha = nn.Parameter(torch.ones((dim, 1)))
        self.bias = nn.Parameter(torch.zeros((dim, 1)))
        self.eps = eps

    def forward(self, x):
        # print(x.size(), self.size)
        avg = x.mean(dim=-2, keepdim=True)
        std = x.std(dim=-2, keepdim=True) + self.eps
        norm = self.alpha * (x - avg) / std + self.bias
        return norm
