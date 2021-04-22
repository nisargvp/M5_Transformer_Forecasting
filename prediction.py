import torch
import numpy as np
from trainOps import DataLoader, compute_loss, get_mask

# set up GPU
device = torch.device("cpu")

# model configuration
CONST_LEN = 28

# load model
# replace x by the epoch number
checkpoint = torch.load('x_checkpoint.pth')
model = checkpoint["model"]
model.load_state_dict(checkpoint["state_dict"])
for parameter in model.parameters():
    parameter.requires_grad = False
model.eval()

# set random seed if used a non default value
dataLoader = DataLoader('valid_X.csv', batch_size=512, cat_exist=True, split=(90, 5, 5))
src_mask, tar_mask = get_mask(4 * CONST_LEN, random=False)
# send src_mask, tar_mask to GPU
src_mask, tar_mask = src_mask.to(device), tar_mask.to(device)
loss_test = []
pred_y = []
mean = torch.Tensor(dataLoader.mean)
std = torch.Tensor(dataLoader.std_)
for i, (cat, x, y) in enumerate(dataLoader.get_test_batch()):
    # print("test mini-batch ", i)
    # send tensors to GPU
    cat, x, y = cat.to(device), x.to(device), y.to(device)
    test_y = model.forward(cat, x, y, src_mask, tar_mask)
    pred_y.append(test_y * std[CONST_LEN:] + mean[:, CONST_LEN:])
    test_loss = compute_loss(test_y, y, tar_mask)
    loss_test.append(test_loss.item())

print("Standardized test dataset loss : ", np.mean(loss_test))


loss_pred = []
for i, (cat, x, y) in enumerate(dataLoader.get_original_test_batch()):
    # print("test mini-batch ", i)
    # send tensors to GPU
    _, _, y = cat.to(device), x.to(device), y.to(device)
    loss_pred.append(compute_loss(pred_y[i], y, tar_mask).item())

print("Original test dataset loss : ", np.mean(loss_pred))

