import torch
import torch.nn as nn

from torch.optim import Adam
from Model import Transformer
from trainOps import compute_loss, DataLoader

# model configuration
seq_len = 28 * 4
channels = [5, 5, 5]
k = 5
dropout = 0.2
model = Transformer(seq_len, channels, k, dropout)

# assume that data is loaded ...
# will implement data loader later
loss_history = []
epoch = 5000
optimizer = Adam(model.parameters(), lr=3e-4)
dataLoader = DataLoader()

for i in range(epoch):
    src, tar = None, None
    src_mask, tar_mask = None, None

    out = model.forward(src, tar, src_mask, tar_mask)
    loss = compute_loss(out, tar, tar_mask)

    # record training loss history
    loss_history.append(loss.item())

    # update parameters using backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


