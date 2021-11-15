
ch = 64
data = {'in_channels': [3] + [item * ch for item in [4, 4, 4]]}

print(data)

import torch
inputs = torch.randn(3,28,28)
print(3*28*28)
print(inputs.nelement())