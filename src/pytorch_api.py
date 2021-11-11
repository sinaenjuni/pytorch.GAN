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

print('GPU state:', torch.cuda.is_available())
isUse_cuda = torch.cuda.is_available()
device = torch.device('cuda' if is_cuda else 'cpu')


###
# Hyper-parameters
###
image_size = 28*28
hidden_size = 256
batch_size = 64
latent_size = 100
start_iters = 0
max_iters = 100000
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
    nn.Linear(hidden_size, 1),
    nn.Sigmoid())

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
d_optimizer = torch.optim.Adam(D.parameters(), lr=0.0002)
g_optimizer = torch.optim.Adam(G.parameters(), lr=0.0002)


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

    real_labels = torch.ones(batch_size, 1).to(device)
    fake_labels = torch.zeros(batch_size, 1).to(device)

    ###
    # Train Discriminator (Optimizer Discriminator)
    ###
    real_outputs = D(real_img)
    d_loss_real = criterion(real_outputs, real_labels)
    # d_loss_real = -1 * torch.log(real_outputs) # -1 for gradient ascending
    real_score = real_outputs

    # print(real_outputs)

    fake_img = G(latent_input)
    fake_outputs = D(fake_img)
    d_loss_fake = criterion(fake_outputs, fake_labels)
    # d_loss_fake = -1 * torch.log(1.-fake_outputs) # -1 for gradient ascending
    fake_score = fake_outputs

    d_loss = d_loss_real + d_loss_fake
    # d_loss = (d_loss_real + d_loss_fake).mean()
    reset_grad()
    d_loss.backward()
    d_optimizer.step()


    ###
    # Train Generator (Optimizer Generator)
    ###

    fake_img = G(latent_input)
    fake_outputs = D(fake_img)
    g_loss = criterion(fake_outputs, real_labels)
    # instead of: torch.log(1.-p_fake).mean() <- explained in Section 3
    # g_loss = -1 * torch.log(fake_outputs).mean()

    reset_grad()
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




