import numpy as np
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor


transforms = Compose([Resize(32),
                      ToTensor(),
                      Normalize(0.5, 0.5),
                      ])

dataset = MNIST("./data", train=True, transform=transforms, download=False)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

print(dataset)
print(loader)

unique = np.unique(dataset.targets)
print(unique)
ind = np.argwhere(dataset.targets == unique[1])
print("ind", ind)
# index = np.where(dataset.targets == unique)
# print("index", index)
# print(unique)
# print(dataset.targets)

