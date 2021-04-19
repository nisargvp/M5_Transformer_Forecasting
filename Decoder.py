import torch.nn as nn
import torch.nn.functional as F

from LayerOps import FeedForward, Norm
from Attention import CategoricalEmbedding, Attention

CONST_CAT_DIM = 3049+7+3+10+3


class DecoderLayer(nn.Module):
    def __init__(self, seq_len, c_in, c_out, k=5, dropout=0.1):
        super().__init__()
        self.h = c_out
        self.norm1 = Norm(c_out)
        self.norm2 = Norm(c_out)
        self.norm3 = Norm(c_out)
        self.norm4 = Norm(c_out)
        self.attn1 = Attention(seq_len, c_out, c_out, k, dropout)
        self.attn2 = Attention(seq_len, c_out, c_out, k, dropout)
        self.ff = FeedForward(c_out, c_out)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.broadcast1 = nn.Conv1d(c_in, c_out, kernel_size=1, bias=False)
        self.broadcast2 = nn.Linear(seq_len, seq_len)

    def forward(self, x, e_output, src_mask, tar_mask):
        if len(x.size()) == 2:
            x.unsqueeze_(0)
            x.transpose_(0, 1)
        assert self.h == e_output.size(1)
        x2 = self.norm1(self.broadcast1(x))
        x = x + self.dropout1(self.attn1(x2, x2, x2, tar_mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.attn1(x2, e_output, e_output, src_mask))
        x2 = self.norm3(x)
        x = x + self.dropout3(self.ff(x2.transpose(-1, -2))).transpose(-1, -2)
        # dim = (batch, c_out, seq_len)
        # broadcast earlier feature vector to later time points
        x = x + F.relu(self.broadcast2(x))
        return self.norm4(x)


class CatDecoder(nn.Module):
    def __init__(self, seq_len, channels=[5,5,5], k=5, dropout=0.1):
        super().__init__()
        self.cat_embed = CategoricalEmbedding(seq_len, channels[0], dropout)
        self.layers = [None]*len(channels)
        channels = [1] + channels
        for i in range(1, len(channels)):
            self.layers[i-1] = DecoderLayer(seq_len, c_in=channels[i-1], c_out=channels[i], k=k, dropout=dropout)
        self.norm = Norm(channels[-1])

    def forward(self, x, e_output, src_mask, tar_mask):
        cat_, x_ = x[0], x[1]
        x_cat = self.cat_embed(cat_)
        x_ = self.layers[0](x_, e_output, src_mask, tar_mask) + x_cat
        for i in range(1, len(self.layers)):
            x_ = self.layers[i](x_, e_output, src_mask, tar_mask)
        return self.norm(x_)






