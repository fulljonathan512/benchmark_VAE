import numpy as np
import torch
import deterministic_gaussian_sampling

mu = torch.arange(0.,3.)
# print(vector.size(0))

# dim = vector.size(0)
# vector = vector.repeat(1,dim,1)
# # vector = vector * vector.transpose(2,1)
# print(vector)

numberOfDiracMixture = 4
std = torch.arange(1.,10.)
print(std)
std = torch.unflatten(std, 0, (3,3))
            
print(std)

# dim = std.shape
# print(dim)
# covr = torch.flatten(std)
# print(covr)
# covr = torch.diag(covr)
# print(covr)
a = torch.tensor(())
for t in std:
  covr = torch.diag(t)
  approx = np.zeros((numberOfDiracMixture, mu.size(0)))
  # print(covr.numpy())
  # print(numberOfDiracMixture)
  # print(mu.size(0))
  # print(approx)
  g2d = deterministic_gaussian_sampling.GaussianToDiracApproximation()
  g2d.approximate_double(covr.numpy(), numberOfDiracMixture, mu.size(0), approx)
  del g2d
  a = torch.cat((a,torch.from_numpy(approx)),0)

print(a)

print("Mu")
mu = torch.arange(0.,9.)
mu = torch.unflatten(mu, 0, (3,3))
print(mu)
mu = torch.repeat_interleave(mu,repeats=numberOfDiracMixture,dim=0)
print(mu)

res = mu + a
print("Res:")
print(res)

# res = mu.repeat(1,numberOfDiracMixture,1) + approx

# print(res)
# print(res.size(0))
