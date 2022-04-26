import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns

import sys
sys.path.append('..')
from utiles.tensorboard import getTensorboard
from utiles.data import getSubDataset
from models.resnet import ResNet18
from models.DCGAN import Generator, Discriminator

name = 'Ensemble/test1_im'
tensorboard_path = f'../../tb_logs/{name}'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

tb = getTensorboard(tensorboard_path)

# Hyper-parameters configuration
num_epochs = 200
batch_size = 64
learning_rate = 0.002
beta1 = 0.5
beta2 = 0.999
ngpu = 1
nz = 100

fixed_noise = torch.randn((64, nz, 1, 1)).to(device)

# sample_dir = '../samples'
# Create a directory if not exists
# if not os.path.exists(sample_dir):
#     os.makedirs(sample_dir)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# Image processing
transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.5, 0.5, 0.5),   # 3 for RGB channels
                                     std=(0.5, 0.5, 0.5))])
#
# transform = transforms.Compose([
#     transforms.Resize(image_size),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5],  # 1 for greyscale channels
#                          std=[0.5])])

# # MNIST dataset
# mnist = torchvision.data_module.MNIST(root='../data/',
#                                    train=True,
#                                    transform=transform,
#                                    download=True)
#
# # Data loader
# data_loader = torch.utils.data.DataLoader(dataset=mnist,
#                                           batch_size=batch_size,
#                                           shuffle=True)


# Dataset define
train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                   train=True,
                                   transform=transform,
                                   download=True)

test_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                   train=False,
                                   transform=transform,
                                   download=True)



# Dataset modify
classes = {'plane':0, 'car':1, 'bird':2, 'cat':3,
           'deer':4, 'dog':5, 'frog':6, 'horse':7, 'ship':8, 'truck':9}
labels = torch.tensor(train_dataset.train_labels)
ratio = [0.5**i for i in range(len(classes))]
# print(classes)
# print(labels)
# print(ratio)
transformed_dataset, count = getSubDataset(dataset=train_dataset,
                                     class_index=classes,
                                     labels=labels,
                                     lratio=ratio)

print(count)
for name, idx in classes.items():
    labels = np.array(test_dataset.test_labels)
    count = np.where(labels == idx)[0]
    print(name, len(count))


# fig = plt.figure(figsize=(9, 6))
# sns.barplot(
#     data=count,
#     x="class",
#     y="original"
# )
# plt.tight_layout()
# tb.add_figure(tag='original_data_dist', figure=fig)
#
#
# # Data loader
# data_loader = torch.utils.data.DataLoader(dataset=transformed_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=True)
#
# # Model define
# G = Generator(ngpu).to(device)
# D = ResNet18()
# D.linear = nn.Linear(512, 1, bias=False)
# D.to(device)
# print(D)
#
#
#
# # custom weights initialization called on netG and netD
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)
#
#
# # D.apply(weights_init)
# # G.apply(weights_init)
#
# d_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(beta1, beta2))
# g_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(beta1, beta2))
#
# def reset_grad():
#     d_optimizer.zero_grad()
#     g_optimizer.zero_grad()
#
#
# # Start training
# total_step = len(data_loader)
# for epoch in range(num_epochs):
#     for i, (images, _) in enumerate(data_loader):
#         # images = images.reshape(batch_size, -1).to(device)
#         batch = images.size(0)
#         images = images.to(device)
#
#         # Create the labels which are later used as input for the BCE loss
#         real_labels = torch.ones((batch,) ).to(device)
#         fake_labels = torch.zeros((batch,) ).to(device)
#
#         # Labels shape is (batch_size, 1): [batch_size, 1]
#
#
#         # ================================================================== #
#         #                      Train the discriminator                       #
#         # ================================================================== #
#
#         # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
#         # Second term of the loss is always zero since real_labels == 1
#         outputs = D(images)
#         outputs = outputs.view(-1)
#         # d_loss_real = criterion(outputs, real_labels)
#         d_loss_real = F.relu(1.-outputs).mean()
#         real_score = outputs
#
#         # Compute BCELoss using fake images
#         # First term of the loss is always zero since fake_labels == 0
#         # z = torch.randn(batch_size, latent_size).to(device) # mean==0, std==1
#         z = torch.randn(batch, nz, 1, 1).to(device) # mean==0, std==1
#         fake_images = G(z)
#
#         outputs = D(fake_images.detach())
#         outputs = outputs.view(-1)
#         # d_loss_fake = criterion(outputs, fake_labels)
#         d_loss_fake = F.relu(1.+outputs).mean()
#         fake_score = outputs
#
#         # Backprop and optimize
#         d_loss = d_loss_real + d_loss_fake
#         reset_grad()
#         d_loss.backward()
#         d_optimizer.step()
#
#         # ================================================================== #
#         #                        Train the generator                         #
#         # ================================================================== #
#
#         # Compute loss with fake images
#         # z = torch.randn(batch_size, latent_size).to(device)
#         fake_images = G(z)
#         outputs = D(fake_images)
#         outputs = outputs.view(-1)
#         gened_score = outputs
#
#         # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
#         # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
#         # g_loss = criterion(outputs, real_labels)
#         g_loss = -outputs.mean()
#
#         # Backprop and optimize
#         reset_grad()
#         g_loss.backward()
#         g_optimizer.step()
#
#         if (i + 1) % 200 == 0:
#             print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
#                   .format(epoch+1, num_epochs, i + 1, total_step, d_loss.item(), g_loss.item(),
#                           real_score.mean().item(), fake_score.mean().item()))
#
#     # Save real images
#     # if (epoch + 1) == 1:
#     #     images = images.reshape(images.size(0), 1, 28, 28)
#     #     save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))
#
#     tb.add_scalars(global_step=epoch + 1,
#                    main_tag='loss',
#                    tag_scalar_dict={'discriminator':d_loss.item(),
#                                     'generator':g_loss.item()})
#
#     tb.add_scalars(global_step=epoch + 1,
#                    main_tag='score',
#                    tag_scalar_dict={'real':real_score.mean().item(),
#                                     'fake':fake_score.mean().item()})
#     # tb.add_scalar(tag='d_loss', global_step=epoch+1, scalar_value=d_loss.item())
#     # tb.add_scalar(tag='g_loss', global_step=epoch+1, scalar_value=g_loss.item())
#     # tb.add_scalar(tag='real_score', global_step=epoch+1, scalar_value=real_score.mean().item())
#     # tb.add_scalar(tag='fake_score', global_step=epoch+1, scalar_value=fake_score.mean().item())
#
#     result_images = denorm(G(fixed_noise))
#     tb.add_images(tag='gened_images', global_step=epoch+1, img_tensor=result_images)
#
#     # Save sampled images
#     # fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
#     # save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch + 1)))
#
#
#
#
#     # Save the model checkpoints
#     SAVE_PATH = f'../../weights/{name}/'
#     if not os.path.exists(SAVE_PATH):
#         os.makedirs(SAVE_PATH)
#     torch.save(G.state_dict(), SAVE_PATH + f'G_{epoch+1}.pth')
#     torch.save(D.state_dict(), SAVE_PATH + f'D_{epoch+1}.pth')