# WGAN CP
# Make critic, Sigmoid activation function of Discriminator remove
# Using RMSprop optimizer with 0.00005 learning rate
# Instead of using BCELoss, WGAN use minmax scalar value
# Clipping discriminator parameters

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
lambda_gp = 10
epochs = 200
learning_rate_g = 0.00005
learning_rate_d = 0.00005
beta1 = 0.5
beta2 = 0.999
sample_dir = '../samples'

fixed_noise = torch.randn(batch_size, noise_dim, 1, 1).to(device)

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
    def __init__(self, nz, nc, ngf):
        super(Generator, self).__init__()

        def layer(in_channel, out_channel, kernel_size, stride, padding, activation):
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel,
                                   kernel_size=kernel_size, stride=stride,
                                   padding=padding, bias=False),
                nn.BatchNorm2d(out_channel),
                activation)

        # self.input_layer = layer(in_channel=nz, out_channel=ngf * 8, kernel_size=4, stride=1, padding=0,
        #                          activation=nn.ReLU(True), use_norm=True)
        self.input_layer = layer(in_channel=nz, out_channel=ngf * 4, kernel_size=4, stride=1, padding=0,
                                 activation=nn.ReLU(True))
        self.layer2 = layer(in_channel=ngf * 4, out_channel=ngf * 2, kernel_size=4, stride=2, padding=1,
                            activation=nn.ReLU(True))
        self.layer3 = layer(in_channel=ngf * 2, out_channel=ngf, kernel_size=4, stride=2, padding=1,
                            activation=nn.ReLU(True))
        self.output_layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=ngf, out_channels=nc, kernel_size=4, stride=2, padding=1),
            nn.Tanh())

    def forward(self, x):
        out = self.input_layer(x)
        # out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.output_layer(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, nc, ndf):
        super(Discriminator, self).__init__()

        def layer(in_channel, out_channel, kernel_size, stride, padding, use_norm, activation):
            if use_norm:
                return nn.Sequential(
                    nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                              kernel_size=kernel_size, stride=stride, padding=padding),
                    nn.BatchNorm2d(out_channel),
                    activation)
            else:
                return nn.Sequential(
                    nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                              kernel_size=kernel_size, stride=stride, padding=padding),
                    activation)

        self.input_layer = layer(in_channel=nc, out_channel=ndf,
                                 kernel_size=4, stride=2, padding=1,
                                 use_norm=False,
                                 activation=nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.layer1 = layer(in_channel=ndf, out_channel=ndf * 2,
                            kernel_size=4, stride=2, padding=1,
                            use_norm=False,
                            activation=nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.layer2 = layer(in_channel=ndf * 2, out_channel=ndf * 4,
                            kernel_size=4, stride=2, padding=1,
                            use_norm=False,
                            activation=nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.output_layer = nn.Sequential(
            nn.Conv2d(in_channels=ndf * 4, out_channels=1, kernel_size=4, stride=1, padding=0),
            # nn.Sigmoid()
        )

    def forward(self, x):
        out = self.input_layer(x)
        out = self.layer1(out)
        out = self.layer2(out)
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


G = Generator(nz=100, ngf=64, nc=1).to(device)
D = Discriminator(nc=1, ndf=64).to(device)

G.apply(weights_init)
D.apply(weights_init)

print(G)
print(D)

# out = G(torch.rand((64, 100, 1, 1)).to(device))
# print(out.size())
#
# out = D(torch.rand(64, 1, 32, 32).to(device))
# print(out.size())

# Binary cross entropy loss and optimizer
mse_loss = nn.MSELoss()
g_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate_g, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate_d, betas=(0.5, 0.999))


def compute_gradient_penalty(D, real_samples, fake_samples):
    # real_samples = real_samples.reshape(real_samples.size(0), 1, 28, 28).to(device)
    # fake_samples = fake_samples.reshape(fake_samples.size(0), 1, 28, 28).to(device)

    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    # alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    alpha = torch.rand(real_samples.size(0), 1, 1, 1).to(device)
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples.data + ((1 - alpha) * fake_samples.data)).requires_grad_(True)
    # d_interpolates = D(interpolates.reshape(real_samples.size(0), -1))
    d_interpolates = D(interpolates)

    weights = torch.ones(d_interpolates.size()).to(device)
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
    gradient_penalty = torch.mean((gradients2L2norm - 1) ** 2)
    return gradient_penalty


# Start training
total_step = len(data_loader)
for epoch in range(epochs):
    for i, (images, _) in enumerate(data_loader):
        _batch = images.size(0)
        # images = images.reshape(_batch, -1).to(device)
        images = images.to(device)

        real_labels = torch.ones(_batch, 1).to(device)
        fake_labels = torch.zeros(_batch, 1).to(device)
        z = torch.randn(_batch, noise_dim, 1, 1).to(device)  # mean==0, std==1

        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        d_optimizer.zero_grad()
        d_output_real = D(images).view(_batch, -1)
        # d_loss_real = mse_loss(outputs, real_labels)
        d_loss_real = -d_output_real.mean()
        real_score = d_output_real

        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0

        fake_images = G(z)
        d_output_fake = D(fake_images.detach()).view(_batch, -1)
        # d_loss_fake = mse_loss(d_output_fake, fake_labels)
        d_loss_fake = d_output_fake.mean()
        fake_score = d_output_fake

        gradient_penalty = compute_gradient_penalty(D, images.data, fake_images.data)
        # Backprop and optimize
        d_loss = 0.5 * (d_loss_real + d_loss_fake) + (lambda_gp * gradient_penalty)
        # reset_grad()
        d_loss.backward()
        d_optimizer.step()


        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #

        # Compute loss with fake images
        g_optimizer.zero_grad()

        # z = torch.randn(_batch, noise_dim).to(device)
        # fake_images = G(z)
        g_output = D(fake_images).view(_batch, -1)
        # g_loss = mse_loss(g_output, real_labels)
        g_loss = -g_output.mean()
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
