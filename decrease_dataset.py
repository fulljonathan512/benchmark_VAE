import numpy as np
import torch

# vector = torch.Tensor([1,2,3,4])
# print(vector.size(0))

# dim = vector.size(0)
# vector = vector.repeat(1,dim,1)
# # vector = vector * vector.transpose(2,1)
# print(vector)

print("start")

loaded_data = np.load("examples/scripts/data/mnist/train_data.npz")
lst = loaded_data.files
print("loaded")

# print(loaded_data)

new_data = []
data_size = 0
new_data_size = 0
print("copy data")
for item in lst:
  # print(item)
  for image in loaded_data[item]:
    data_size = data_size + 1
    if(data_size % 100 == 0):
      # print(image)
      new_data_size = new_data_size + 1
      new_data.append(image)
#   # print(loaded_data[item])

np.savez("examples/scripts/data/mnist/new_train_data.npz", **{"data":new_data})
print(f"new data set created with size {new_data_size}")

# new_loaded_data = np.load("new_mnist.npz")

# print(new_loaded_data)
# lst = new_loaded_data.files
# for item in lst:
#   print(item)
#   for image in new_loaded_data[item]:
#     print(image)

# print(counter)