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
n_critic = 5
lambda_gp = 10
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

def compute_gradient_penalty(D, real_samples, fake_samples):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.rand((real_samples.size())).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)

    weights = torch.ones(real_samples.size()).to(device)
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=weights,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    gradients = gradients.view(gradients.size(0), -1)
    gradients2L2norm = torch.sqrt(torch.sum(gradients ** 2, dim=1))
    gradient_penalty = torch.mean(( gradients2L2norm - 1 ) ** 2)
    return gradient_penalty


for i in range(start_iters, max_iters):
    iter_loader = iter(loader)
    real_img, _ = iter_loader.next()
    real_img = real_img.reshape(-1, image_size).to(device)

    latent_input = getLatentVector(batch_size).to(device)

    ###
    # Train Discriminator (Optimizer Discriminator)
    ###ㅎ

    real_outputs = D(real_img)
    real_score = real_outputs

    fake_img = G(latent_input)
    fake_outputs = D(fake_img)
    fake_score = fake_outputs

    gradient_penalty = compute_gradient_penalty(D, real_img.data, fake_img.data)
    d_loss = -torch.mean(real_outputs) + torch.mean(fake_outputs) + lambda_gp * gradient_penalty

    d_optimizer.zero_grad()
    d_loss.backward()
    d_optimizer.step()


    if i % n_critic == 0:

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




