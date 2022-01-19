# Discriminator not use Sigmoid activation function so discriminator named as critic
# Not using Binary cross entropy loss, just minmax output scare value
# Using RMSProp optimizer
# Clamping weights for Discriminator (-0.01, 0.01)
# Training Generator per n_critic


import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from utiles.imbalance_mnist_loader import ImbalanceMNISTDataLoader
import matplotlib.pyplot as plt
from functools import reduce

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# TensorBoard define
# log_dir = '../../tb_logs/vanillaGAN/test3'
# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)
# tb = SummaryWriter(log_dir=log_dir)

# Hyper-parameters
image_size = (1, 32, 32)
noise_dim = 100
hidden_size = 256
batch_size = 64

epochs = 200
learning_rate_g = 0.00005
learning_rate_d = 0.00005
beta1 = 0.5
beta2 = 0.999
n_critic = 5
sample_dir = '../samples'

fixed_noise = torch.randn(batch_size, noise_dim).to(device)

# Create a directory if not exists
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)


def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)


# Image processing
# transform = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=(0.5, 0.5, 0.5),   # 3 for RGB channels
#                                      std=(0.5, 0.5, 0.5))])

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],  # 1 for greyscale channels
                         std=[0.5])])

# MNIST dataset
mnist = torchvision.datasets.MNIST(root='../../data/',
                                   train=True,
                                   transform=transform,
                                   download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=batch_size,
                                          shuffle=True)


# Generator
class Generator(nn.Module):
    def __init__(self, nz, image_size):
        super(Generator, self).__init__()
        output_size = reduce(lambda x, y: x * y, image_size)

        def layer(in_channel, out_channel, activation):
            return nn.Sequential(nn.Linear(in_features=in_channel, out_features=out_channel),
                                 nn.BatchNorm1d(out_channel, 0.8),
                                 activation)

        self.input_layer = nn.Sequential(nn.Linear(in_features=nz, out_features=128), nn.ReLU(True))
        self.layer1 = layer(in_channel=128, out_channel=256, activation=nn.ReLU(True))
        self.layer2 = layer(in_channel=256, out_channel=512, activation=nn.ReLU(True))
        self.layer3 = layer(in_channel=512, out_channel=1024, activation=nn.ReLU(True))
        self.output_layer = nn.Sequential(nn.Linear(in_features=1024, out_features=output_size), nn.Tanh())

    def forward(self, x):
        out = self.input_layer(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.output_layer(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, image_size, nc):
        super(Discriminator, self).__init__()
        input_size = reduce(lambda x, y: x * y, image_size)

        def layer(in_channel, out_channel, activation):
            return nn.Sequential(nn.Linear(in_features=in_channel, out_features=out_channel),
                                 nn.BatchNorm1d(out_channel),
                                 activation)

        self.input_layer = nn.Sequential(
            nn.Linear(in_features=input_size, out_features=512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.layer1 = layer(in_channel=512, out_channel=256,
                            activation=nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.output_layer = nn.Sequential(
            nn.Linear(in_features=256, out_features=nc))

    def forward(self, x):
        out = self.input_layer(x)
        out = self.layer1(out)
        out = self.output_layer(out)
        return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
    elif classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


G = Generator(nz=100, image_size=image_size).to(device)
D = Discriminator(image_size=image_size, nc=1).to(device)

# G.apply(weights_init)
# D.apply(weights_init)

print(G)
print(D)

# Binary cross entropy loss and optimizer
# bce_loss = nn.BCELoss()
# g_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate_g, betas=(0.5, 0.999))
# d_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate_d, betas=(0.5, 0.999))

g_optimizer = torch.optim.RMSprop(G.parameters(), lr=learning_rate_g)
d_optimizer = torch.optim.RMSprop(D.parameters(), lr=learning_rate_d)

# Start training
total_step = len(data_loader)
for epoch in range(epochs):
    for i, (images, _) in enumerate(data_loader):
        _batch = images.size(0)
        images = images.reshape(_batch, -1).to(device)

        real_labels = torch.ones(_batch, 1).to(device)
        fake_labels = torch.zeros(_batch, 1).to(device)

        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        d_optimizer.zero_grad()
        d_output_real = D(images)
        # d_loss_real = bce_loss(outputs, real_labels)
        d_loss_real = -torch.mean(d_output_real)
        real_score = d_output_real

        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = torch.randn(_batch, noise_dim).to(device)  # mean==0, std==1
        fake_images = G(z)

        d_output_fake = D(fake_images.detach())
        # d_loss_fake = bce_loss(d_output_fake, fake_labels)
        d_loss_fake = torch.mean(d_output_fake)
        fake_score = d_output_fake

        # Backprop and optimize
        d_loss = 0.5 * (d_loss_real + d_loss_fake)
        # reset_grad()
        d_loss.backward()
        d_optimizer.step()


        # clipping D
        for p in D.parameters():
            p.data.clamp_(-0.01, 0.01)


        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #
        if i % n_critic == 0:
            # Compute loss with fake images
            g_optimizer.zero_grad()

            # z = torch.randn(_batch, noise_dim).to(device)
            # fake_images = G(z)
            g_output = D(fake_images)
            # g_loss = bce_loss(g_output, real_labels)
            g_loss = -torch.mean(g_output)
            gened_score = g_output

            # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
            # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf

            # Backprop and optimize
            # reset_grad()
            g_loss.backward()
            g_optimizer.step()

        if (i + 1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                  .format(epoch + 1, epochs, i + 1, total_step, d_loss.item(), g_loss.item(),
                          real_score.mean().item(), fake_score.mean().item()))

    result_images = denorm(G(fixed_noise)).detach().cpu()
    result_images = result_images.reshape(result_images.size(0), 1, 32, 32)
    result_images = make_grid(result_images).permute(1, 2, 0)
    # print(result_images.size())
    plt.imshow(result_images.numpy())
    plt.show()

    # Save real images
    # if (epoch + 1) == 1:
    #     images = images.reshape(images.size(0), 1, 28, 28)
    #     save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))

    # tb.add_scalar(tag='d_loss', global_step=epoch+1, scalar_value=d_loss.item())
    # tb.add_scalar(tag='g_loss', global_step=epoch+1, scalar_value=g_loss.item())
    # tb.add_scalar(tag='real_score', global_step=epoch+1, scalar_value=real_score.mean().item())
    # tb.add_scalar(tag='fake_score', global_step=epoch+1, scalar_value=fake_score.mean().item())
    #
    # result_images = denorm(G(fixed_noise))
    # result_images = result_images.reshape(result_images.size(0), 1, 28, 28)
    # tb.add_images(tag='gened_images', global_step=epoch+1, img_tensor=result_images)

    # Save sampled images
    # fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    # save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch + 1)))

# Save the model checkpoints
# torch.save(G.state_dict(), 'G.ckpt')
# torch.save(D.state_dict(), 'D.ckpt')
