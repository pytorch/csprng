import torch

from torchcsprng._C import *

def random_key_tensor(size, dtype, device, gen):
    t = torch.empty(size, dtype=dtype, device='cpu')
    return fill_random_key_tensor(t, gen).to(device)
