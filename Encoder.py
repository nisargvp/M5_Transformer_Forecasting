import torch.nn as nn

from LayerOps import FeedForward, Norm
from Attention import CategoricalEmbedding, Attention

CONST_CAT_DIM = 3049+7+3+10+3


class EncoderLayer(nn.Module):
    def __init__(self, seq_len, c_in, c_out, k=5, dropout=0.1):
        super().__init__()
        self.norm1 = Norm(c_out)
        self.norm2 = Norm(c_out)
        self.attn = Attention(seq_len, c_in=c_in, c_out=c_out, k=k)
        self.broadcast = nn.Conv1d(c_in, c_out, kernel_size=1, bias=False)
        self.ff = FeedForward(c_out, c_out)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.norm1(self.broadcast(x))
        x = x + self.dropout1(self.attn(x, x, x, mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.ff(x2))
        return x


class CatEncoder(nn.Module):
    def __init__(self, seq_len, channels=[5,5,5], k=5, dropout=0.1):
        super().__init__()
        self.cat_embed = CategoricalEmbedding(seq_len, channels[0], dropout)
        self.layers = [None] * len(channels)
        channels = [1] + channels
        for i in range(1, len(channels)):
            self.layers[i] = EncoderLayer(seq_len,  c_in=channels[i-1], c_out=channels[i], k=k, dropout=dropout)
        self.norm = Norm(channels[-1])

    def forward(self, x, mask):
        cat_, x_ = x[:, :CONST_CAT_DIM], x[:, CONST_CAT_DIM:]
        x_cat = self.cat_embed(cat_)
        x_ = self.layers[0](x_, mask) + x_cat
        for i in range(1, len(self.layers)):
            x_ = self.layers[i](x_, mask)
        return self.norm(x)







