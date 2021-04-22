"""
* A test.py for understanding masking
* Won't be used in any training module
"""

import torch
from torch.autograd import Variable
import numpy as np


def subsequent_mask(size):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    s_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(s_mask) == 0


class Batch:
    "Object for holding a batch of data with mask during training."

    def __init__(self, src, trg=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if trg is not None:
            self.trg = trg[:, :-1]
            self.trg_y = trg[:, 1:]
            self.trg_mask = self.make_std_mask(self.trg, pad)
            self.ntokens = (self.trg_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & Variable(
            subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data))
        return tgt_mask


def data_gen(V, batch, nbatches):
    "Generate random data for a src-tgt copy task."
    for i in range(nbatches):
        data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))
        data[:, 0] = 1
        src = Variable(data, requires_grad=False)
        print(src)
        tgt = Variable(data, requires_grad=False)
        print(tgt)
        yield Batch(src, tgt, 0)