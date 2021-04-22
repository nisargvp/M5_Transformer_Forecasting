import torch
import numpy as np
import matplotlib.pyplot as plt

from torch.optim import Adam
from Model import Transformer
from trainOps import compute_loss, DataLoader, get_mask, create_small_dataset, check_tensor

# torch.autograd.set_detect_anomaly(True)
# set up GPU
device = torch.device("cpu")

# model configuration
CONST_LEN = 28
seq_len = 28 * 4
channels = [5, 5, 5]
conv_k = 5
dropout = 0.3

# load model
# replace x by the epoch number
checkpoint = torch.load('10_checkpoint.pth')
model = checkpoint["model"]
# load model from the checkpoint
model.load_state_dict(checkpoint["state_dict"])
# send model to GPU
model.to(device)

# load optimizer from the checkpoint
optimizer = Adam(model.parameters(), lr=3e-4)
optimizer.load_state_dict(checkpoint["optimizer"])
# training code
loss_train_history = []
loss_valid_history = []
epoch = 200
# create_small_dataset(data_file="valid_X.csv", csv_name="small_X.csv")
dataLoader = DataLoader('small_X.csv', batch_size=16, cat_exist=True, split=(90, 5, 5))

src_mask, tar_mask = get_mask(4 * CONST_LEN, random=False)
# send src_mask, tar_mask to GPU
src_mask, tar_mask = src_mask.to(device), tar_mask.to(device)

print("re-start a previous training ... ")

for k in range(epoch):

    if k and k % 50 == 0:
        checkpoint = {'model': Transformer(seq_len, channels, k, dropout),
                      'state_dict': model.state_dict(),
                      'optimizer' : optimizer.state_dict()}
        torch.save(checkpoint, str(k)+'_'+'checkpoint_re.pth')

    loss_train = []
    dataLoader.shuffle()
    # set model training state
    model.train()
    for i, (cat, src, tar) in enumerate(dataLoader.get_training_batch()):
        # print("train mini-batch ", i)
        # send tensors to GPU
        # print("train - check input: ", check_tensor([cat, src, tar]))
        cat, src, tar = cat.to(device), src.to(device), tar.to(device)
        # print(src.size())
        out = model.forward(cat, src, tar, src_mask, tar_mask)
        # print("train - check out: ", check_tensor([out]))
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
    loss_valid = []
    with torch.no_grad():
        model.eval()
        for i, (cat, x, y) in enumerate(dataLoader.get_validation_batch()):
            # print("validation mini-batch ", i)
            # send tensors to GPU
            # print("validation - check input: ", check_tensor([cat, src, tar]))
            cat, x, y = cat.to(device), x.to(device), y.to(device)
            valid_y = model.forward(cat, x, y, src_mask, tar_mask)
            valid_loss = compute_loss(valid_y, y, tar_mask)
            # print("train - check out: ", check_tensor([valid_loss]))
            loss_valid.append(valid_loss.item())

    loss_valid_history.append(np.mean(loss_valid))

    print("epoch:", k,
          "training loss = ", loss_train_history[-1],
          "validation loss = ", loss_valid_history[-1])

plt.plot(list(range(1, epoch+1)), loss_train_history, label='train')
plt.plot(list(range(1, epoch+1)), loss_valid_history, label='valid')
plt.xlabel('epoch')
plt.ylabel('MSE loss')
plt.legend()
plt.savefig('loss_plot.png')


