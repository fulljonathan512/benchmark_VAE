import numpy as np
import torch

vector = torch.Tensor([1,2,3,4])
print(vector.size(0))

dim = vector.size(0)
vector = vector.repeat(1,dim,1)
# vector = vector * vector.transpose(2,1)
print(vector)