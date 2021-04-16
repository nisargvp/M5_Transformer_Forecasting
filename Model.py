import torch.nn as nn
import torch.nn.functional as F

from Encoder import CatEncoder
from Decoder import CatDecoder


class Transformer(nn.Module):
    def __init__(self, seq_len, channels=[5,5,5], k=5, dropout=0.1):
        self.encoder = CatEncoder(seq_len, channels, k, dropout)
        self.decoder = CatDecoder(seq_len, channels, k, dropout)
        self.out = nn.Linear(channels[-1], 1, dropout)

    def forward(self, x, tar, src_mask, tar_mask):
        e_output = self.encoder(x, src_mask)
        d_output = self.decoder(x, tar, src_mask, tar_mask)
        output = F.relu(self.out(d_output))
        return output