import torch, math
import torch.nn as nn
import torch.nn.functional as F

CONST_CAT_DIM= 3049+7+3+10+3  # item + depart + cat + store + state


def _attention(q, k, v, d_k, mask=None, dropout=None):
    # d_k = embed dim
    scores = torch.matmul(q.transpose(-2, -1), k) / math.sqrt(d_k)
    if mask is not None:
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e9)
    scores = F.softmax(scores, dim=-1)
    if dropout is not None:
        scores = dropout(scores)
    # output dim = (batch, heads / channel / embed dim, seq_len)
    output = torch.matmul(scores, v.transpose(-2, -1))
    return output


# Attention Module should be used after CategoricalAttention Module
# after categorical information has been encoded
class Attention(nn.Module):
    def __init__(self, seq_len, c_in=1, c_out=1, k=5, dropout=0.1):
        super().__init__()
        # model dimension ?
        self.seq_len = seq_len
        self.d_model = seq_len * c_out
        # dimension per head
        self.h = c_out
        # reflection padding, only do reflection padding on the left (causal)
        # 04/20/2021 a bug that hasn't been fixed
        # ref: https://github.com/pytorch/pytorch/issues/49601
        # ref: https://github.com/pytorch/pytorch/issues/55222
        self.padding = nn.ReflectionPad1d((k-1, 0))
        # kernel size used for local + causal convolution
        self.k = k
        # input is padded by reflection
        self.q_conv = nn.Conv1d(c_in, c_out, kernel_size=k, stride=1, bias=False)
        self.v_conv = nn.Conv1d(c_in, c_out, kernel_size=k, stride=1, bias=False)
        self.k_conv = nn.Conv1d(c_in, c_out, kernel_size=k, stride=1, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(c_out, c_out)

    def forward(self, q, k, v, mask=None):
        # batch_size = q.size(0)
        # print(q.size())
        # reflection padding input sequence, 2nd part of input
        # get seq embedding
        q_seq_embed = self.q_conv(self.padding(q))
        # q_seq_embed = self.q_conv(q)
        v_seq_embed = self.v_conv(self.padding(k))
        # v_seq_embed = self.v_conv(k)
        k_seq_embed = self.k_conv(self.padding(v))
        # k_seq_embed = self.k_conv(v)
        # scores dim = (batch, seq_len, heads)
        scores = _attention(q_seq_embed, k_seq_embed, v_seq_embed, self.h, mask, self.dropout)

        return self.out(scores).transpose(-2, -1)


class CategoricalEmbedding(nn.Module):
    def __init__(self, seq_len, c_out, dropout=0.1):
        super().__init__()
        self.seq_len = seq_len
        self.h = c_out
        self.cat_embed = nn.Linear(CONST_CAT_DIM, seq_len * c_out, dropout)

    def forward(self, x):
        batch_size = x.size(0)
        # print(x.size())
        # get cat embedding and reshape it
        x_cat_embed = self.cat_embed(x)
        return x_cat_embed.view(batch_size, self.h, self.seq_len)