import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import sys
sys.path.append('..')
from utiles.tensorboard import getTensorboard
from models.resnet import ResNet18
from models.cDCGAN import Discriminator, Generator
from utiles.dataset import CIFAR10, MNIST

name = 'cDCGAN/mnist_test1'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# TensorBoard define
tensorboard_path = f'../../tb_logs/{name}'
tb = getTensorboard(log_dir=tensorboard_path)

# Hyper-parameters
nc = 1
ndf = 32

nz = 100
ngf = 32
ncls = 10

image_size = 32
batch_size = 128

num_epochs = 200
learning_rate = 0.0002
beta1 = 0.5
beta2 = 0.999
ngpu = 1


# fixed data for generator input
g_fixed_label = torch.zeros(ncls*10, ncls)
g_fixed_label_index = torch.LongTensor([ i//10 for i in range(100) ]).view(ncls*10, 1)
g_fixed_label = g_fixed_label.scatter_(1, g_fixed_label_index, 1).view(ncls*10, ncls, 1, 1).to(device)

g_fixed_noise = torch.randn((10, 100, 1, 1)).repeat(10, 1, 1, 1).to(device)
print(g_fixed_noise.size())
print(g_fixed_label.size())

# label format for training generator
label2Glabel = torch.zeros((ncls, ncls))
g_label_index = torch.LongTensor([ i for i in range(10) ]).view(ncls, 1)
label2Glabel = label2Glabel.scatter_(1, g_label_index, 1).view(ncls, ncls, 1, 1).to(device)


# sample_dir = '../samples'
# Create a directory if not exists
# if not os.path.exists(sample_dir):
#     os.makedirs(sample_dir)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# Dataset modify
dataset = MNIST(32)
# dataset = CIFAR10()
train_dataset = dataset.getTrainDataset()
transformed_dataset, count = dataset.getTransformedDataset([0.5**i for i in range(len(dataset.classes))])
test_dataset = dataset.getTestDataset()

print(train_dataset)
print("count", count["transformed"])

# ce_weights = [1-(i/sum(count["transformed"])) for i in count["transformed"]]
# ce_weights = torch.FloatTensor(ce_weights).to(device)
# print(ce_weights)


fig = plt.figure(figsize=(9, 6))
sns.barplot(
    data=count,
    x="class",
    y="original"
)
plt.tight_layout()
tb.add_figure(tag='original_data_dist', figure=fig)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)



# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# Device setting
D = Discriminator(nc=nc, ndf=ndf).to(device)
G = Generator(nz=nz, nc=nc, ngf=ngf, ncls=ncls).to(device)

# temp_in = torch.randn((10,3,32,32)).to(device)
# for i in D.modules():
#     if 'Conv2d' == i.__class__.__name__:
#         y = i(temp_in)
#         print(y.size())
#         temp_in = y
# print('\n\n')
# temp_in = torch.randn((10,100,1,1)).to(device)
# for i in G.modules():
#     if 'ConvTranspose2d' == i.__class__.__name__:
#         y = i(temp_in)
#         print(y.size())
#         temp_in = y

D.apply(weights_init)
G.apply(weights_init)

# SAVE_PATH = f'../../weights/DCGAN/test1/'
# G.load_state_dict(torch.load(SAVE_PATH + 'G_200.pth'))
# D.load_state_dict(torch.load(SAVE_PATH + 'D_200.pth'))

# Binary cross entropy loss and optimizer
# criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(beta1, beta2))
g_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(beta1, beta2))

# d_optimizer = torch.optim.RMSprop(D.parameters(), lr = 0.00005)#, weight_decay=opt.weight_decay_D)
# g_optimizer = torch.optim.RMSprop(G.parameters(), lr = 0.00005, weight_decay=0)


def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()


# Start training
total_step = len(data_loader)
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(data_loader):
        # images = images.reshape(batch_size, -1).to(device)
        batch = images.size(0)
        images = images.to(device)
        labels = labels.to(device)

        # Create the labels which are later used as input for the BCE loss
        # real_labels = torch.ones((batch,) ).to(device)
        # fake_labels = torch.zeros((batch,) ).to(device)

        # Labels shape is (batch_size, 1): [batch_size, 1]


        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        out_adv, out_cls = D(images)

        out_adv = out_adv.squeeze()
        out_cls = out_cls.squeeze()
        # d_loss_real = criterion(outputs, real_labels)

        d_loss_adv = F.relu(1.-out_adv).mean()
        d_loss_cls = F.cross_entropy(out_cls, labels)
        d_loss_real = .5 * (d_loss_adv + d_loss_cls)

        real_score = out_adv
        real_cls_loss = d_loss_cls

        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        # z = torch.randn(batch_size, latent_size).to(device) # mean==0, std==1
        c_ = (torch.rand(batch, 1) * ncls).long().squeeze().to(device) # 균등한 확률로 0~1사이의 랜덤
        cls = label2Glabel[c_].to(device)

        z = torch.randn(batch, nz, 1, 1).to(device) # mean==0, std==1

        fake_images = G(z, cls)

        out_adv, out_cls = D(fake_images.detach())
        out_adv = out_adv.squeeze()
        out_cls = out_cls.squeeze()

        # d_loss_fake = criterion(outputs, fake_labels)
        d_loss_adv = F.relu(1.+out_adv).mean()
        d_loss_cls = F.cross_entropy(out_cls, labels)
        d_loss_fake = .5 * (d_loss_adv + d_loss_cls)

        fake_score = out_adv
        fake_cls_loss = d_loss_cls

        d_loss = d_loss_real + d_loss_fake

        # Backprop and optimize
        reset_grad()
        d_loss.backward()
        d_optimizer.step()

        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #



        # Compute loss with fake images
        # z = torch.randn(batch_size, latent_size).to(device)
        z = torch.randn(batch, nz, 1, 1).to(device) # mean==0, std==1
        c_ = (torch.rand(batch, 1) * ncls).long().squeeze().to(device)
        cls = label2Glabel[c_].to(device)
        fake_images = G(z, cls)

        out_adv, out_cls = D(fake_images)
        out_adv = out_adv.squeeze()
        out_cls = out_cls.squeeze()

        g_loss_adv = -out_adv.mean()
        g_loss_cls = F.cross_entropy(out_cls, labels)

        gened_score = out_adv


        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        # g_loss = criterion(outputs, real_labels)
        g_loss = .5 * (g_loss_adv + g_loss_cls)

        # Backprop and optimize
        reset_grad()
        g_loss.backward()
        g_optimizer.step()

        if (i + 1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                  .format(epoch+1, num_epochs, i + 1, total_step, d_loss.item(), g_loss.item(),
                          real_score.mean().item(), fake_score.mean().item()))

    # Save real images
    # if (epoch + 1) == 1:
    #     images = images.reshape(images.size(0), 1, 28, 28)
    #     save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))

    tb.add_scalars(global_step=epoch + 1,
                   main_tag='loss',
                   tag_scalar_dict={'discriminator':d_loss.item(),
                                    'generator':g_loss.item()})

    tb.add_scalars(global_step=epoch + 1,
                   main_tag='score',
                   tag_scalar_dict={'real':real_score.mean().item(),
                                    'fake':fake_score.mean().item()})

    # tb.add_scalar(tag='d_loss', global_step=epoch+1, scalar_value=d_loss.item())
    # tb.add_scalar(tag='g_loss', global_step=epoch+1, scalar_value=g_loss.item())
    # tb.add_scalar(tag='real_score', global_step=epoch+1, scalar_value=real_score.mean().item())
    # tb.add_scalar(tag='fake_score', global_step=epoch+1, scalar_value=fake_score.mean().item())

    result_images = denorm(G(g_fixed_noise, g_fixed_label))
    tb.add_images(tag='gened_images', global_step=epoch+1, img_tensor=result_images)

    # Save sampled images
    # fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    # save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch + 1)))


    # # Save the model checkpoints
    # SAVE_PATH = f'../../weights/{name}/'
    # if not os.path.exists(SAVE_PATH):
    #     os.makedirs(SAVE_PATH)
    # torch.save(G.state_dict(), SAVE_PATH + f'G_{epoch+1}.pth')
    # torch.save(D.state_dict(), SAVE_PATH + f'D_{epoch+1}.pth')