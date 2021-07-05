import torch
from torch.nn import Module, Sequential
from torch.utils.data import DataLoader

import torchvision
from torchvision.utils import make_grid, save_image
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print(torch.cuda.is_available())


batch_size = 16

###
# Prepare MNIST dataset
###

transform = Compose([ToTensor(), Normalize([.5], [.5])])

DATAPATH = Path.home() / 'dataset'
dataset = MNIST(root=DATAPATH, train=True, download=True, transform=transform)
print(dataset[0][0].min(), dataset[0][0].max())


loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
iters = iter(loader)
img, label = iters.next()

# class Generator(Module):
#     def __init__(self):
#         self.main = Sequential([])
#
#     def forward(self, x):





# save_image(img, './1.png')

# print(img.size())
# grid = make_grid(img)
# print(type(grid))
# plt.imshow(grid.permute(1,2,0))
# plt.show()


# print(np.asarray(dataset[1]))
# plt.imshow(dataset[1][0])
# plt.show()




