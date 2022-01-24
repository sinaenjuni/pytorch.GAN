import numpy as np
import torch
from torchvision.datasets import MNIST, CIFAR10
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor
from torchvision.utils import make_grid
import matplotlib.pyplot as plt


transforms = Compose([Resize(32),
                      ToTensor(),
                      Normalize(0.5, 0.5),
                      ])

dataset = CIFAR10("./data", train=True, transform=transforms, download=False)
loader = DataLoader(dataset, batch_size=100, shuffle=True)


targets = dataset.targets
# print(targets)

unique = np.unique(targets)

ind = [np.where(targets == unique[i])[0][0] for i in unique]
print("ind", ind)

images_per_class = dataset.data[ind] / 255
images_per_class = torch.tensor(images_per_class)

images_per_class = images_per_class.permute(0,3,1,2)
images_per_class = make_grid(images_per_class, normalize=True, nrow=10).permute(1,2,0).contiguous()
print(images_per_class[0])
plt.imshow(images_per_class)
plt.show()


iter = iter(loader).__next__()[0]

image_grid = make_grid(iter, normalize=True, nrow=10).permute(1,2,0).contiguous()
image_grid = torch.cat([images_per_class, image_grid], 0)
print(image_grid.size())
plt.imshow(image_grid)
plt.show()