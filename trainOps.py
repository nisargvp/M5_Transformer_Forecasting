import torch
import pandas as pd
from numpy.random import randint, shuffle

CONST_LEN = 28
CONST_CAT_DIM = 3049+7+3+10+3


def compute_loss(y, pred, mask):
    batch = y.size(0)
    # print(y.size())
    # print(pred.size())
    diff = y - pred
    mask = 1 - mask.unsqueeze(-1)
    return torch.sum(torch.matmul(diff**2, mask)) / (torch.sum(mask) * batch)


def get_mask(seq_len=4*CONST_LEN, random=False):
    src_mask = [1]*seq_len
    tar_mask = [1]*seq_len
    if not random:
        tar_mask[-CONST_LEN:] = [0] * CONST_LEN
    else:
        pos = randint(0, seq_len-1, size=1)
        tar_mask[pos:] = [0] * (seq_len - pos)
    return torch.Tensor(src_mask), torch.Tensor(tar_mask)


def create_small_dataset(data_file, csv_name="small_X.csv", size=1000):
    dat = pd.read_csv(data_file)
    n, _ = dat.shape
    # categorical variables, numerical variables
    cat = dat.iloc[:, :5]
    cat = pd.concat([pd.get_dummies(cat.iloc[:, j]) for j in range(5)], axis=1)
    # print(cat.shape)
    cat_x = pd.concat([cat.iloc[:size, :], dat.iloc[:size, 5:]], axis=1)
    cat_x.to_csv(csv_name, index=False)
    print("A small dataset was created!")


class DataLoader:
    def __init__(self, data_file, batch_size=10, cat_exist=False, split=(8, 1, 1)):

        dat = pd.read_csv(data_file)
        self.n, _ = dat.shape
        # print("dataset size : ", self.n)
        self.batch_size = batch_size
        self.batch = self.n // batch_size
        self.train_n = round(self.batch * split[0] / sum(split))
        # print("train_n = ", self.train_n)
        self.valid_n = round(self.batch * split[1] / sum(split))
        self.test_n = self.batch - self.train_n - self.valid_n
        # random shuffle dataset rows
        # then do train/valid/test split
        # categorical variables, numerical variables
        dat = dat.sample(frac=1).reset_index(drop=True)
        if not cat_exist:
            cat, self.dat = dat.iloc[:, :5], dat.iloc[:, 5:]
            self.cat = pd.concat([pd.get_dummies(cat.iloc[:, j]) for j in range(5)], axis=1)
        else:
            self.cat, self.dat = dat.iloc[:, :CONST_CAT_DIM], dat.iloc[:, CONST_CAT_DIM:]

        assert self.cat.shape[1] == CONST_CAT_DIM

        self.train_dat = self.dat.iloc[:self.train_n*batch_size, :]
        mean = self.train_dat.mean(axis=0)
        std = self.train_dat.std(axis=0)
        std.replace(0.0, 1.0, inplace=True)
        self.mean = mean.tolist()
        self.std = std.tolist()
        self.train_dat = (self.train_dat - mean) / std
        self.train_cat = self.cat.iloc[:self.train_n*batch_size, :]
        # validation dataset
        self.valid_dat = self.dat.iloc[self.train_n*batch_size:(self.train_n + self.valid_n)*batch_size, :]
        self.valid_dat = (self.valid_dat - mean) / std
        self.valid_cat = self.cat.iloc[self.train_n*batch_size:(self.train_n + self.valid_n)*batch_size, :]
        # test dataset
        self.test_dat = self.dat.iloc[(self.train_n + self.valid_n)*batch_size:, :]
        self.test_dat = (self.test_dat - mean) / std
        self.test_cat = self.cat.iloc[(self.train_n + self.valid_n)*batch_size:, :]
        # print(self.train_n, self.valid_n, self.test_n)

    def shuffle(self):
        # training dataset
        train_size = self.train_dat.shape[0]
        new_order = list(range(train_size))
        shuffle(new_order)
        self.train_dat = self.train_dat.iloc[new_order, :]
        self.train_cat = self.train_cat.iloc[new_order, :]

    def get_training_batch(self):

        for i in range(1, self.train_n):
            l = self.train_cat.iloc[((i-1)*self.batch_size):(i*self.batch_size), :]
            x = self.train_dat.iloc[((i-1)*self.batch_size):(i*self.batch_size), :(4*CONST_LEN)]
            y = self.train_dat.iloc[((i-1)*self.batch_size):(i*self.batch_size), CONST_LEN:]
            # print(l.shape, x.shape, y.shape)
            yield torch.Tensor(l.to_numpy()), torch.Tensor(x.to_numpy()), torch.Tensor(y.to_numpy())

    def get_validation_batch(self):

        for i in range(1, self.valid_n):
            l = self.valid_cat.iloc[((i - 1) * self.batch_size):(i * self.batch_size), :]
            x = self.valid_dat.iloc[((i - 1) * self.batch_size):(i * self.batch_size), :(4 * CONST_LEN)]
            y = self.valid_dat.iloc[((i - 1) * self.batch_size):(i * self.batch_size), CONST_LEN:]
            # print(l.shape, x.shape, y.shape)
            yield torch.Tensor(l.to_numpy()), torch.Tensor(x.to_numpy()), torch.Tensor(y.to_numpy())

    def get_test_batch(self):

        for i in range(1, self.test_n):
            l = self.test_cat.iloc[((i - 1) * self.batch_size):(i * self.batch_size), :]
            x = self.test_dat.iloc[((i - 1) * self.batch_size):(i * self.batch_size), :(4 * CONST_LEN)]
            y = self.test_dat.iloc[((i - 1) * self.batch_size):(i * self.batch_size), CONST_LEN:]
            # print(l.shape, x.shape, y.shape)
            yield torch.Tensor(l.to_numpy()), torch.Tensor(x.to_numpy()), torch.Tensor(y.to_numpy())