import torch.nn as nn

from LayerOps import FeedForward, Norm
from Attention import CategoricalEmbedding, Attention

CONST_CAT_DIM = 3049+7+3+10+3


class DecoderLayer(nn.Module):
    def __init__(self, seq_len, c_in, c_out, k=5, dropout=0.1):
        super().__init__()
        self.norm1 = Norm(c_out)
        self.norm2 = Norm(c_out)
        self.norm3 = Norm(c_out)
        self.attn1 = Attention(seq_len, c_in, c_out, k, dropout)
        self.attn2 = Attention(seq_len, c_in, c_out, k, dropout)
        self.ff = FeedForward(c_out, c_out)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, e_output, x_mask, tar_mask):
        # figure it out
        # why tar_mask is used in self-attention
        # while x_mask is used in cross attention
        x2 = self.norm1(x)
        x = x + self.dropout1(self.attn1(x2, x2, x2, tar_mask))
        x2 = self.norm2(x)
        x = x + self.dropout2(self.attn1(x2, e_output, e_output, x_mask))
        x2 = self.norm3(x)
        x = x + self.dropout3(self.ff(x2))
        return x


class CatDecoder(nn.Module):
    def __init__(self, seq_len, channels=[5,5,5], k=5, dropout=0.1):
        super().__init__()
        self.cat_embed = CategoricalEmbedding(seq_len, channels[0], dropout)
        self.layers = [None]*len(channels)
        channels = [1] + channels
        for i in range(1, len(channels)):
            self.layers[i] = DecoderLayer(seq_len, heads=channels[i], c_in=channels[i-1], k=k, dropout=dropout)
        self.norm = Norm(channels[-1])

    def forward(self, x, e_output, src_mask, tar_mask):
        cat_, x_ = x[:, :CONST_CAT_DIM], x[:, CONST_CAT_DIM:]
        x_cat = self.cat_embed(cat_)
        x_ = self.layers[0](x_, e_output, src_mask, tar_mask) + x_cat
        for i in range(1, len(self.layers)):
            x_ = self.layers[i](x_, e_output, src_mask, tar_mask)
        return self.norm(x)






