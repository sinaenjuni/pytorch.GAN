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
from utiles.dataset import CIFAR10, MNIST

name = 'DCGAN/cifar10_test1_50img'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# TensorBoard define
tensorboard_path = f'../../tb_logs/{name}'
tb = getTensorboard(log_dir=tensorboard_path)

# Hyper-parameters
nc=3
nz=100
ngf=32
ndf=32

image_size = 32
batch_size = 64

num_epochs = 200 * 10
learning_rate = 0.0002
beta1 = 0.5
beta2 = 0.999
ngpu = 1


fixed_noise = torch.randn((64, nz, 1, 1)).to(device)

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
transformed_dataset, count = dataset.getTransformedDataset([0.01 for i in range(len(dataset.classes))])
test_dataset = dataset.getTestDataset()

print(train_dataset)
print(count)

fig = plt.figure(figsize=(9, 6))
sns.barplot(
    data=count,
    x="class",
    y="original"
)
plt.tight_layout()
tb.add_figure(tag='original_data_dist', figure=fig)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=transformed_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


# G(z)
class Generator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(Generator, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(100, d*2, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(d*2)
        self.deconv1_2 = nn.ConvTranspose2d(10, d*2, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(d*2)
        self.deconv2 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*2)
        self.deconv3 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d)
        self.deconv4 = nn.ConvTranspose2d(d, 1, 4, 2, 1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))
        y = F.relu(self.deconv1_2_bn(self.deconv1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.tanh(self.deconv4(x))
        # x = F.relu(self.deconv4_bn(self.deconv4(x)))
        # x = F.tanh(self.deconv5(x))

        return x

class Discriminator(nn.Module):
    # initializers
    def __init__(self, d=128):
        super(Discriminator, self).__init__()
        self.conv1_1 = nn.Conv2d(1, d/2, 4, 2, 1)
        self.conv1_2 = nn.Conv2d(10, d/2, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d*2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d*2)
        self.conv3 = nn.Conv2d(d*2, d*4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d*4)
        self.conv4 = nn.Conv2d(d * 4, 1, 4, 1, 0)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = F.leaky_relu(self.conv1_1(input), 0.2)
        y = F.leaky_relu(self.conv1_2(label), 0.2)
        x = torch.cat([x, y], 1)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = F.sigmoid(self.conv4(x))

        return x

# Device setting
D = Discriminator(ngpu).to(device)
G = Generator(ngpu).to(device)

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

# D.apply(weights_init)
# G.apply(weights_init)

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
    for i, (images, _) in enumerate(data_loader):
        # images = images.reshape(batch_size, -1).to(device)
        batch = images.size(0)
        images = images.to(device)

        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones((batch,) ).to(device)
        fake_labels = torch.zeros((batch,) ).to(device)

        # Labels shape is (batch_size, 1): [batch_size, 1]


        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        outputs = D(images)
        outputs = outputs.view(-1)
        # d_loss_real = criterion(outputs, real_labels)
        d_loss_real = F.relu(1.-outputs).mean()
        real_score = outputs

        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        # z = torch.randn(batch_size, latent_size).to(device) # mean==0, std==1
        z = torch.randn(batch, nz, 1, 1).to(device) # mean==0, std==1
        fake_images = G(z)

        outputs = D(fake_images.detach())
        outputs = outputs.view(-1)
        # d_loss_fake = criterion(outputs, fake_labels)
        d_loss_fake = F.relu(1.+outputs).mean()
        fake_score = outputs

        # Backprop and optimize
        d_loss = d_loss_real + d_loss_fake
        reset_grad()
        d_loss.backward()
        d_optimizer.step()

        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #

        # Compute loss with fake images
        # z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        outputs = outputs.view(-1)
        gened_score = outputs

        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        # g_loss = criterion(outputs, real_labels)
        g_loss = -outputs.mean()

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


    # Save the model checkpoints
    SAVE_PATH = f'../../weights/{name}/'
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
    torch.save(G.state_dict(), SAVE_PATH + f'G_{epoch+1}.pth')
    torch.save(D.state_dict(), SAVE_PATH + f'D_{epoch+1}.pth')