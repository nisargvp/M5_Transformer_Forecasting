import torch


def compute_loss(y, pred, mask):
    batch = y.size(0)
    diff = y - pred
    mask = 1 - torch.Tensor(mask).unsqueeze(-1)
    return torch.sum(torch.matmul(diff**2, mask)) / (torch.sum(mask) * batch)


# implement it later
def DataLoader():
    pass
