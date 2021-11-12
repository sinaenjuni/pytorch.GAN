import torch

batch = 128
real_label = 1

tensor = torch.full((batch,), real_label)

print(tensor)