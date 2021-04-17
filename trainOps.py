import torch
import pandas as pd
from numpy.random import randint

CONST_LEN = 28


def compute_loss(y, pred, mask):
    batch = y.size(0)
    diff = y - pred
    mask = 1 - torch.Tensor(mask).unsqueeze(-1)
    return torch.sum(torch.matmul(diff**2, mask)) / (torch.sum(mask) * batch)


def get_mask(seq_len=4*CONST_LEN, random=False):
    src_mask = [1]*seq_len
    tar_mask = [1]*seq_len
    if not random:
        tar_mask[-CONST_LEN:] = [0] * CONST_LEN
    else:
        pos = randint(0, seq_len-1, size=1)
        tar_mask[pos:] = [0] * (seq_len - pos)
    return src_mask, tar_mask


class DataLoader:
    def __init__(self, data_file, batch_size=10, split=(8, 1, 1)):
        dat = pd.read_csv(data_file)
        self.n, _ = dat.shape
        self.batch_size = batch_size
        self.batch = self.n // batch_size
        self.valid_n = self.batch // sum(split) * split[1]
        self.test_n = self.batch // sum(split) * split[2]
        self.train_n = self.batch - self.valid_n - self.test_n

        # random shuffle dataset rows
        # then do train/valid/test split
        dat = dat.sample(frac=1).reset_index(drop=True)
        self.train_dat = dat.iloc[:self.train_n, :]
        self.valid_dat = dat.iloc[self.train_n:(self.train_n+self.valid_n), :]
        self.test_dat = dat.iloc[(self.train_n+self.valid_n):, :]

    def get_training_batch(self):

        for i in range(1, self.train_n):
            x = self.train_dat.iloc[(i-1)*self.batch_size:(i)*self.batch_size, :(4*CONST_LEN)]
            y = self.train_dat.iloc[(i-1)*self.batch_size:(i)*self.batch_size, CONST_LEN:]
            yield x.to_numpy(), y.to_numpy()

    def get_validation_batch(self):
        x = self.valid_dat.iloc[:, :(4*CONST_LEN)]
        y = self.valid_dat.iloc[:, CONST_LEN:]
        return x.to_numpy(), y.to_numpy()

    def get_test_batch(self):
        x = self.test_dat.iloc[:, :(4 * CONST_LEN)]
        y = self.test_dat.iloc[:, CONST_LEN:]
        return x.to_numpy(), y.to_numpy()