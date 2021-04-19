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
        super().__init__()
        self.encoder = CatEncoder(seq_len, channels, k, dropout)
        self.decoder = CatDecoder(seq_len, channels, k, dropout)
        self.out = nn.Conv1d(channels[-1], 1, stride=1, kernel_size=1)

    def forward(self, cat, x, tar, src_mask, tar_mask):
        # print("cat size: ", cat.size())
        # print("x size: ", x.size())
        # print("tar size: ", tar.size())
        e_output = self.encoder([cat, x], src_mask)
        d_output = self.decoder([cat, tar], e_output, src_mask, tar_mask)
        # print(d_output.size())
        # predict the full sequence using d_output: (batch, c_out, seq_len)
        output = F.selu(self.out(d_output))
        return output
