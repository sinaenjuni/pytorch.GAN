import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import to_pil_image

import torchvision
from torchvision import transforms
from torchvision.utils import make_grid, save_image

from utiles.imbalance_mnist import IMBALANCEMNIST
from utiles.imbalance_cifar import IMBALANCECIFAR10
import matplotlib.pyplot as plt
from functools import reduce
import numpy as np
from utiles.sampler import SelectSampler

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# TensorBoard define
# log_dir = '../../tb_logs/vanillaGAN/test3'
# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)
# tb = SummaryWriter(log_dir=log_dir)

# Save dir define
weights_dir = './weights/ACGAN_balancedCE/ '
if not os.path.exists(weights_dir):
    os.makedirs(weights_dir)

# Hyper-parameters
image_size = (1, 32, 32)
noise_dim = 100
hidden_size = 256
batch_size = 60
num_class = 10
epochs = 200
learning_rate_g = 0.0002
learning_rate_d = 0.0002
beta1 = 0.5
beta2 = 0.999
sample_dir = '../samples'

target_label = 6

# fixed_noise = torch.randn(10, noise_dim, 1, 1).to(device).repeat(10, 1, 1, 1)
# fixed_noise = torch.randn(100, noise_dim, 1, 1).to(device)
fixed_noise = torch.randn(batch_size, noise_dim, 1, 1).to(device)


# index = torch.tensor([[i % 10] for i in range(100)])
# fixed_onehot = torch.zeros(100, 10).scatter_(1, index, 1).to(device).view(100, 10, 1, 1)

index = torch.tensor([[target_label] for _ in range(batch_size)])
fixed_onehot = torch.zeros(batch_size, 10).scatter_(1, index, 1).to(device).view(batch_size, 10, 1, 1)

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
# mnist = torchvision.data_module.MNIST(root='../../data/',
#                                    train=True,
#                                    transform=transform,
#                                    download=True)

dataset_train = IMBALANCEMNIST(root='../../data/',
                           train=True,
                           transform=transform,
                           download=False,
                           imb_factor=0.01)

dataset_test = IMBALANCEMNIST(root='../../data/',
                           train=False,
                           transform=transform,
                           download=False,
                              imb_factor=1)

# dataset = IMBALANCECIFAR10(root='../../data/',
#                            train=True,
#                            transform=transform,
#                            download=True,
#                            imb_factor=0.01)

# # Data loader
# select_sampler_train = SelectSampler(data_source=dataset_train, target_label=target_label, shuffle=False)
# loader_train = torch.utils.data.DataLoader(dataset=dataset_train,
#                                           batch_size=batch_size,
#                                           sampler=select_sampler_train)
#
# select_sampler_test = SelectSampler(data_source=dataset_test, target_label=target_label, shuffle=False)
# loader_test = torch.utils.data.DataLoader(dataset=dataset_test,
#                                           batch_size=batch_size,
#                                           sampler=select_sampler_test)


# print(dataset_train.get_cls_num_list())
# print(dataset_train.num_per_cls_dict)
#
# cls_num_list = torch.tensor(dataset_train.get_cls_num_list()).to(device)
# prior = cls_num_list / torch.sum(cls_num_list)
# inverse_prior, _ = torch.sort(prior, descending=False)
#
# print('prior', prior)
# print('inverse_prior', inverse_prior)


# Generator
class Generator(nn.Module):
    def __init__(self, nz, nc, ngf, num_class):
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
        self.input_layer = layer(in_channel=nz + num_class, out_channel=ngf * 4, kernel_size=4, stride=1, padding=0,
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
    def __init__(self, nc, ndf, num_class):
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
        self.adv_layer = nn.Sequential(
            nn.Conv2d(in_channels=ndf * 4, out_channels=1, kernel_size=4, stride=1, padding=0),
            nn.Sigmoid())

        self.cls_layer = nn.Sequential(
            nn.Conv2d(in_channels=ndf * 4, out_channels=num_class, kernel_size=4, stride=1, padding=0))


    def forward(self, x):
        out = self.input_layer(x)
        out = self.layer1(out)
        out = self.layer2(out)
        adv = self.adv_layer(out)
        cls = self.cls_layer(out)
        return adv, cls


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


G = Generator(nz=100, ngf=64, nc=1, num_class=num_class).to(device)
D = Discriminator(nc=1, ndf=64, num_class=num_class).to(device)


bce_loss = nn.BCELoss()
ce_loss = nn.CrossEntropyLoss()

adv_real_labels = torch.ones(batch_size, 1).to(device)
cls_target_labels = torch.zeros(batch_size, dtype=torch.long).fill_(target_label).to(device)

select_sampler_train = SelectSampler(data_source=dataset_train, target_label=target_label, shuffle=False)
loader_train = torch.utils.data.DataLoader(dataset=dataset_train,
                                          batch_size=batch_size,
                                          sampler=select_sampler_train)

select_sampler_test = SelectSampler(data_source=dataset_test, target_label=target_label, shuffle=False)
loader_test = torch.utils.data.DataLoader(dataset=dataset_test,
                                          batch_size=batch_size,
                                          sampler=select_sampler_test)


print(f"Epoch\t"
      f"Train adv\t"
      f"Train cls\t"
      f"Test adv\t"
      f"Test cls\t"
      f"Gened adv\t"
      f"Gened cls")

for i in range(200):
    print(f"{i}", end='\t')
    G.load_state_dict(torch.load(f"./weights/ACGAN_balancedCE/ G_{i}.ckpt"))
    D.load_state_dict(torch.load(f"./weights/ACGAN_balancedCE/ D_{i}.ckpt"))

    train_data = iter(loader_train).__next__()[0].to(device)
    test_data = iter(loader_test).__next__()[0].to(device)
    gened_data = G(torch.cat([fixed_noise, fixed_onehot], dim=1)).detach()

    images_dir = f'./images/ACGAN_balancedCE/{target_label}/'
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    save_image(make_grid(gened_data.detach().cpu(), nrow=10, normalize=True), fp=images_dir+f"{i}.png")
    # grid = save_image(gened_data.detach().cpu(), nrow=10, normalize=True).permute(1,2,0).contiguous()


    output_adv, logit_cls = D(train_data)
    adv_loss = bce_loss(output_adv.view(-1, 1), adv_real_labels)
    cls_loss = ce_loss(logit_cls.view(-1, 10), cls_target_labels)
    print(f"{adv_loss.item():.4}\t"
          f"{cls_loss.item():.4}", end='\t')


    output_adv, logit_cls = D(test_data)
    adv_loss = bce_loss(output_adv.view(-1, 1), adv_real_labels)
    cls_loss = ce_loss(logit_cls.view(-1, 10), cls_target_labels)
    print(f"{adv_loss.item():.4}\t"
          f"{cls_loss.item():.4}", end='\t')


    output_adv, logit_cls = D(gened_data)
    adv_loss = bce_loss(output_adv.view(-1, 1), adv_real_labels)
    cls_loss = ce_loss(logit_cls.view(-1, 10), cls_target_labels)
    print(f"{adv_loss.item():.4}\t"
          f"{cls_loss.item():.4}")


