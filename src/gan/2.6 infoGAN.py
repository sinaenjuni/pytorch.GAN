import os
import torch
import numpy as np
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
num_z = 62
num_dis_c = 1
dis_c_dim = 10
num_con_c = 2

hidden_size = 256
batch_size = 64

epochs = 200
learning_rate_g = 0.0002
learning_rate_d = 0.0002
beta1 = 0.5
beta2 = 0.999
sample_dir = '../samples'



z = torch.randn(100, num_z, 1, 1)
fixed_noise = z

idx = torch.tensor([[i // 10] for i in range(100)])
dis_c = torch.zeros(100, dis_c_dim).scatter_(1, idx, 1.0).view(100, dis_c_dim, 1, 1)
con_c = torch.rand(100, num_con_c, 1, 1) * 2 - 1

fixed_noise = torch.cat([fixed_noise, dis_c, con_c], dim=1)


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


class GaussianNLLLoss(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, c, c_hat, sigma):
        assert sigma > 0
        l = (c - c_hat) ** 2
        l /= (2 * sigma ** 2)
        l += torch.log(sigma)
        return l.mean()

def getNoiseSample(dis_c_dim, num_con_c, num_z, batch_size):
    z = torch.randn(batch_size, num_z, 1, 1)

    idx = (torch.rand((batch_size, 1)) * dis_c_dim).type(torch.long)
    dis_c = torch.zeros(batch_size, dis_c_dim).scatter_(1, idx, 1.0).view(batch_size, dis_c_dim, 1, 1)
    con_c = torch.rand(batch_size, num_con_c, 1, 1) * 2 - 1

    noise = torch.cat([z, dis_c, con_c], dim = 1)
    return noise.to(device), idx.to(device)

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
        self.output_layer = nn.Conv2d(in_channels=ndf * 4,
                                      out_channels=ndf * 4 * 4,
                                      kernel_size=4,
                                      stride=1,
                                      padding=0)
    def forward(self, x):
        out = self.input_layer(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.output_layer(out)
        return out

class DHead(nn.Module):
    def __init__(self):
        super(DHead, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(1024, 1, 1),
                                  nn.Sigmoid())
    def forward(self, x):
        return self.conv(x)

class QHead(nn.Module):
    def __init__(self):
        super(QHead, self).__init__()
        self.conv = nn.Sequential(nn.Conv2d(1024, 128, 1, bias=False),
                                  nn.BatchNorm2d(128),
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True))
        self.conv_disc = nn.Conv2d(128, 10, 1)
        self.conv_mu = nn.Conv2d(128, 2, 1)
        self.conv_var = nn.Conv2d(128, 2, 1)
    def forward(self, x):
        x = self.conv(x)

        disc_logit = self.conv_disc(x)
        mu = self.conv_mu(x)
        var = self.conv_var(x)

        return disc_logit, mu, var


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


G = Generator(nz=74, ngf=64, nc=1).to(device)
D = Discriminator(nc=1, ndf=64).to(device)

netD = DHead().to(device)
netQ = QHead().to(device)
netD.apply(weights_init)
netQ.apply(weights_init)

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
bce_loss = nn.BCELoss()
ce_loss = nn.CrossEntropyLoss()
nnll_loss = GaussianNLLLoss()

g_optimizer = torch.optim.Adam([{'params': G.parameters(),
                                 'params': netQ.parameters()}], lr=learning_rate_g, betas=(0.5, 0.999))
d_optimizer = torch.optim.Adam([{'params': D.parameters(),
                                 'params': netD.parameters()}], lr=learning_rate_d, betas=(0.5, 0.999))

# Start training
total_step = len(data_loader)
for epoch in range(epochs):
    for i, (images, _) in enumerate(data_loader):
        _batch = images.size(0)
        # images = images.reshape(_batch, -1).to(device)
        images = images.to(device)

        real_labels = torch.ones(_batch, 1).to(device)
        fake_labels = torch.zeros(_batch, 1).to(device)
        # z = torch.randn(_batch, noise_dim, 1, 1).to(device)  # mean==0, std==1

        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        d_optimizer.zero_grad()

        logits_d_real = D(images)
        outputs_d_real = netD(logits_d_real).view(_batch, -1)
        d_loss_real = bce_loss(outputs_d_real, real_labels)
        real_score = outputs_d_real

        z, idx = getNoiseSample(dis_c_dim, num_con_c, num_z, _batch)
        fake_images = G(z)
        logits_d_fake = D(fake_images.detach())
        outputs_d_fake = netD(logits_d_fake).view(_batch, -1)
        d_loss_fake = bce_loss(outputs_d_fake, fake_labels)
        fake_score = outputs_d_fake

        # Backprop and optimize
        d_loss = d_loss_real + d_loss_fake
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
        logits_g = D(fake_images)
        outputs_g = netD(logits_g).view(_batch, -1)
        g_loss = bce_loss(outputs_g, real_labels)
        gened_score = outputs_g

        q_logits, q_mu, q_var = netQ(logits_g)
        dis_loss = 0
        for j in range(num_dis_c):
            dis_loss += ce_loss(q_logits[:, j * 10: j * 10 + 10], idx[j])

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
