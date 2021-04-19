import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.optim import Adam
from Model import Transformer
from trainOps import compute_loss, DataLoader, get_mask, create_small_dataset

# model configuration
CONST_LEN = 28
seq_len = 28 * 4
channels = [5, 5, 5]
k = 5
dropout = 0.2
model = Transformer(seq_len, channels, k, dropout)

# assume that data is loaded ...
# will implement data loader later
loss_train_history = []
loss_valid_history = []
epoch = 10
optimizer = Adam(model.parameters(), lr=3e-4)
# create_small_dataset(data_file="valid_X.csv", csv_name="small_X.csv")
dataLoader = DataLoader('valid_X.csv', batch_size=256, cat_exist=False, split=(90, 5, 5))

src_mask, tar_mask = get_mask(4 * CONST_LEN, random=False)

for i in range(epoch):

    if i and i % 500 == 0:
        checkpoint = {'model': Transformer(seq_len, channels, k, dropout),
                      'state_dict': model.state_dict(),
                      'optimizer' : optimizer.state_dict()}
        torch.save(checkpoint, str(i)+'_'+'checkpoint.pth')

    loss_train = []
    dataLoader.shuffle()
    # set model training state
    model.train()
    for cat, src, tar in dataLoader.get_training_batch():
        # print(src.size())
        out = model.forward(cat, src, tar, src_mask, tar_mask)
        # print(out.size())
        loss = compute_loss(out, tar, tar_mask)

        # record training loss history
        loss_train.append(loss.item())

        # update parameters using backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    loss_train_history.append(np.mean(loss_train))

    # model evaluation mode
    with torch.no_grad():
        model.eval()
        cat, x, y = dataLoader.get_validation_batch()
        valid_y = model.forward(cat, x, y, src_mask, tar_mask)
        loss_valid = compute_loss(valid_y, y, tar_mask)

    loss_valid_history.append(loss_valid.item())

    print("epoch:", i,
          "training loss = ", loss_train_history[-1],
          "validation loss = ", loss_valid_history[-1])

plt.plot(list(range(1, epoch+1)), loss_train_history, label='train')
plt.plot(list(range(1, epoch+1)), loss_valid_history, label='valid')
plt.xlabel('epoch')
plt.ylabel('MSE loss')
plt.legend()
plt.savefig('loss_plot.png')


