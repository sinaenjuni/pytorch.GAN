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
# from models.resnet import ResNet18
# from models.cDCGAN import Discriminator, Generator
from models.ACGAN import Discriminator, Generator
from utiles.dataset import CIFAR10, MNIST

name = 'ACGAN/cifar_test1'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# TensorBoard define
tensorboard_path = f'../../tb_logs/{name}'
tb = getTensorboard(log_dir=tensorboard_path)

# Hyper-parameters
nc = 3
nz = 100
ngf = 38
ndf = 16
ncls = 10

image_size = 32
batch_size = 128

num_epochs = 200
learning_rate = 0.0002
beta1 = 0.5
beta2 = 0.999
ngpu = 1


# fixed_noise = torch.randn((64, nz, 1, 1)).to(device)
fixed_noise = torch.randn((64, nz)).to(device)

# fixed noise & label
# temp_z_ = torch.randn(10, 100)
# fixed_z_ = temp_z_
# fixed_y_ = torch.zeros(10, 1)
# for i in range(9):
#     fixed_z_ = torch.cat([fixed_z_, temp_z_], 0)
#     temp = torch.ones(10, 1) + i
#     fixed_y_ = torch.cat([fixed_y_, temp], 0)
#
# fixed_z_ = fixed_z_.view(-1, 100, 1, 1).to(device)
#
# fixed_y_label_ = torch.zeros(100, 10)
# fixed_y_label_.scatter_(1, fixed_y_.type(torch.LongTensor), 1)
# fixed_y_label_ = fixed_y_label_.view(-1, 10, 1, 1).to(device)
#
#
# onehot = torch.zeros(ncls, ncls)
# onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
#                          .view(ncls, 1), 1).view(ncls, ncls, 1, 1)

# sample_dir = '../samples'
# Create a directory if not exists
# if not os.path.exists(sample_dir):
#     os.makedirs(sample_dir)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# Dataset modify
# dataset = MNIST(32)
dataset = CIFAR10()
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
D = Discriminator(nc=nc, ncls=ncls, nf=ndf).to(device)
G = Generator(nz=nz, nc=nc, nf=ngf).to(device)

D.apply(weights_init)
G.apply(weights_init)

# SAVE_PATH = f'../../weights/DCGAN/test1/'
# G.load_state_dict(torch.load(SAVE_PATH + 'G_200.pth'))
# D.load_state_dict(torch.load(SAVE_PATH + 'D_200.pth'))

# Binary cross entropy loss and optimizer
adv_criterion = nn.BCELoss()
cls_criterion = nn.NLLLoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate, betas=(beta1, beta2))
g_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate, betas=(beta1, beta2))


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
        real_labels = torch.ones((batch,)).to(device)
        fake_labels = torch.zeros((batch,)).to(device)

        # Labels shape is (batch_size, 1): [batch_size, 1]


        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        out_adv, out_cls = D(images)

        out_adv = out_adv.view(-1)
        out_cls = out_cls.view(-1, 10)
        # d_loss_real = criterion(outputs, real_labels)

        d_loss_adv = F.relu(1.-out_adv).mean()
        d_loss_cls = F.nll_loss(out_cls, labels)
        d_loss_real = 0.5 * (d_loss_adv + d_loss_cls)

        real_score = out_adv
        real_cls_loss = d_loss_cls

        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        # z = torch.randn(batch_size, latent_size).to(device) # mean==0, std==1
        # c_ = (torch.rand(batch, 1) * ncls).long().squeeze().to(device) # 균등한 확률로 0~1사이의 랜덤
        # cls = onehot[c_].to(device)
        z = torch.randn(batch, nz).to(device) # mean==0, std==1
        gened_labels = (torch.rand((batch,))*10).long().to(device)

        fake_images = G(z)

        out_adv, out_cls = D(fake_images.detach())
        out_adv = out_adv.view(-1)
        out_cls = out_cls.view(-1, 10)

        # d_loss_fake = criterion(outputs, fake_labels)
        d_loss_adv = F.relu(1.+out_adv).mean()
        d_loss_cls = F.nll_loss(out_cls, gened_labels)
        d_loss_fake = 0.5 * (d_loss_adv + d_loss_cls)

        fake_score = out_adv
        fake_cls_loss = d_loss_cls

        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        # Backprop and optimize
        reset_grad()
        d_loss.backward()
        d_optimizer.step()

        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #

        # Compute loss with fake images
        # z = torch.randn(batch_size, latent_size).to(device)
        z = torch.randn(batch, nz).to(device) # mean==0, std==1
        gened_labels = (torch.rand((batch,))*10).long().to(device)

        fake_images = G(z)
        out_adv, out_cls = D(fake_images)
        out_adv = out_adv.view(-1)
        out_cls = out_cls.view(-1, 10)

        g_loss_adv = -out_adv.mean()
        g_loss_cls = F.nll_loss(out_cls, gened_labels)

        gened_score = out_adv


        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        # g_loss = criterion(outputs, real_labels)
        g_loss = 0.5 * (g_loss_adv + g_loss_cls)

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

    result_images = denorm(G(fixed_noise))
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