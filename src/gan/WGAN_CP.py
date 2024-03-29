import torch
import torch.nn as nn
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
isUse_cuda = torch.cuda.is_available()
device = torch.device('cuda' if isUse_cuda else 'cpu')


###
# Hyper-parameters
###
image_size = 28*28
hidden_size = 256
batch_size = 64
latent_size = 100
start_iters = 0
max_iters = 100000
clip_value = 0.01
n_critic = 5

sample_dir = 'samples'

DATAPATH = Path.home() / 'dataset'
SAVEPATH = Path('sample_dir')

if not SAVEPATH.exists():
    SAVEPATH.mkdir(exist_ok=True, parents=True)


###
# Prepare MNIST dataset
###

transform = Compose([ToTensor(), Normalize([.5], [.5])])
dataset = MNIST(root=DATAPATH, train=True, download=True, transform=transform)
print(dataset[0][0].min(), dataset[0][0].max())


loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
iters = iter(loader)
img, label = iters.next()

###
# Define Models
###

# Discriminator
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(),
    nn.Dropout(0.1),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(),
    nn.Dropout(0.1),
    nn.Linear(hidden_size, 1)
    # nn.Sigmoid()
)

# Generator
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(hidden_size, image_size),
    nn.Tanh())

# Device setting
D = D.to(device)
G = G.to(device)


# Binary cross entropy loss and optimizer
criterion = nn.BCELoss()
d_optimizer = torch.optim.RMSprop(D.parameters(), lr=0.00005)
g_optimizer = torch.optim.RMSprop(G.parameters(), lr=0.00005)

# momentum 계열의 optimizer를 사용하면 학습이 불안정하다.


def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


def getLatentVector(batch_size):
    return torch.randn(batch_size, latent_size)


for i in range(start_iters, max_iters):
    iter_loader = iter(loader)
    real_img, _ = iter_loader.next()
    real_img = real_img.reshape(-1, image_size).to(device)

    latent_input = getLatentVector(batch_size).to(device)

    ###
    # Train Discriminator (Optimizer Discriminator)
    ###

    real_outputs = D(real_img)
    real_score = real_outputs

    fake_img = G(latent_input)
    fake_outputs = D(fake_img)
    fake_score = fake_outputs

    d_loss = -torch.mean(real_outputs) + torch.mean(fake_outputs)

    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()

    # Clip weights of discriminator
    for p in D.parameters():
        p.data.clamp_(-clip_value, clip_value)

    if (i+1) % n_critic == 0:

        ###
        # Train Generator (Optimizer Generator)
        ###

        fake_img = G(latent_input)
        fake_outputs = D(fake_img)

        g_loss = -torch.mean(fake_outputs)

        g_optimizer.zero_grad()
        g_loss.backward()
        g_optimizer.step()

    if (i+1) % 1000 == 0:
        print('iters [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
              .format(i+1, max_iters,
                      d_loss.item(), g_loss.item(),
                      real_score.mean().item(), fake_score.mean().item()))

        generated_img = fake_img.reshape(batch_size, 1, 28, 28).cpu()
        grid = make_grid(denorm(generated_img))
        plt.imshow(grid.permute(1,2,0))
        plt.show()



# save_image(img, './1.png')

# print(img.size())
# grid = make_grid(img)
# print(type(grid))
# plt.imshow(grid.permute(1,2,0))
# plt.show()


# print(np.asarray(dataset[1]))
# plt.imshow(dataset[1][0])
# plt.show()




