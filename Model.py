import torch.nn as nn
import torch.nn.functional as F

from Encoder import CatEncoder
from Decoder import CatDecoder


"""
The data dimension is always organized as: (batch, features / c_out, seq_len)
For the initial input, the dim = (batch, seq_len), where the second dim is missing 
                       (or we can treat the second dim = 1)  
"""


class Transformer(nn.Module):
    def __init__(self, seq_len, channels=[5,5,5], k=5, dropout=0.1):
        self.encoder = CatEncoder(seq_len, channels, k, dropout)
        self.decoder = CatDecoder(seq_len, channels, k, dropout)
        self.out = nn.Conv1d(channels[-1], 1)

    def forward(self, x, tar, src_mask, tar_mask):
        e_output = self.encoder(x, src_mask)
        d_output = self.decoder(tar, e_output, src_mask, tar_mask)
        # predict the full sequence using d_output: (batch, c_out, seq_len)
        output = F.relu(self.out(d_output))
        return output
