import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns

from torchsummaryX import summary

from utiles.tensorboard import getTensorboard
from utiles.data import getSubDataset
from utiles.imbalance_cifar10_loader import ImbalanceCIFAR10DataLoader
from models.resnet_s import resnet32
from models.cDCGAN import Generator


# Define hyper-parameters
name = 'prop5/test1'
tensorboard_path = f'../../tb_logs/{name}'

num_workers = 4
num_epochs = 800
batch_size = 128

learning_rate = 0.001
weight_decay = 5e-4
momentum = 0.9
nesterov = True


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# Define Tensorboard
tb = getTensorboard(tensorboard_path)

# Define DataLoader
train_data_loader = ImbalanceCIFAR10DataLoader(data_dir='../../data',
                                              batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers, training=True, imb_factor=0.01)

test_data_loader = ImbalanceCIFAR10DataLoader(data_dir='../../data',
                                              batch_size=batch_size,
                                              shuffle=False, num_workers=num_workers, training=False)


print("Number of train dataset", len(train_data_loader.dataset))
print("Number of test dataset", len(test_data_loader.dataset))

# Define hyper-parameters
name = 'prop5/test1'
tensorboard_path = f'../../tb_logs/{name}'

num_workers = 4
num_epochs = 800
batch_size = 128

learning_rate = 0.001
weight_decay = 5e-4
momentum = 0.9
nesterov = True

beta1 = 0.5
beta2 = 0.999

nz = 100
nc = 3
ncls = 10
ngf = 128
ndf = 64

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




# Define Discriminator
class Propose(nn.Module):
    def __init__(self, back_born_model, in_channels=32, ndf=32):
        super(Propose, self).__init__()
        self.back_born_model = back_born_model

        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=ndf,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=ndf,
                      out_channels=ndf*2,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=ndf*2,
                      out_channels=ndf*4,
                      kernel_size=3,
                      stride=2,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(in_channels=ndf*4,
                      out_channels=1,
                      kernel_size=3,
                      stride=1,
                      padding=1,
                      bias=False),
        )

    def forward(self, inputs):
        classifier = self.back_born_model(inputs)
        discriminator_input = inputs

        for idx, layer in enumerate(self.back_born_model.children()):
            if idx == 4:
                break
            discriminator_input = layer(discriminator_input)
            # print(idx)
        # print(discriminator_input.size())

        discriminator = self.discriminator(discriminator_input)
        discriminator = discriminator.squeeze()

        return classifier, discriminator


# Define model
model = resnet32(num_classes=10, use_norm=True).to(device)
proposed = Propose(model, in_channels=32, ndf=ndf).to(device)
generator = Generator(nz, nc, ncls, ngf).to(device)

# Output test
# input_tensor = torch.rand((32,3,32,32)).to(device)
# classifier, discriminator = proposed(input_tensor)
# print(classifier.size(), discriminator.size())


proposed_optimizer = torch.optim.Adam(proposed.parameters(), lr=learning_rate, betas=(beta1, beta2))
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, beta2))

def reset_grad():
    proposed_optimizer.zero_grad()
    generator_optimizer.zero_grad()


test_best_accuracy = 0
test_best_accuracy_epoch = 0

log = {}

# Start training
total_step = len(train_data_loader)
for epoch in range(num_epochs):
    test_accuracy = 0
    for i, (images, labels) in enumerate(train_data_loader):
        # images = images.reshape(batch_size, -1).to(device)
        batch = images.size(0)
        images = images.to(device)
        labels = labels.to(device)

        proposed.train()
        generator.train()

        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        out_cls, out_adv = proposed(images)

        d_loss_adv = F.relu(1.-out_adv).mean()
        d_loss_cls = F.cross_entropy(out_cls, labels)
        d_loss_real = (d_loss_adv + d_loss_cls)

        log['D_real_adv'] = d_loss_adv.item()
        log['D_real_cls'] = d_loss_cls.item()
        log['D_real_loss'] = d_loss_real.item()


        #===============================
        c_ = (torch.rand(batch, 1) * ncls).long().squeeze().to(device)  # 균등한 확률로 0~1사이의 랜덤
        cls = label2Glabel[c_].to(device)
        z = torch.randn(batch, nz, 1, 1).to(device)  # mean==0, std==1
        fake_images = generator(z, cls)

        out_cls, out_adv = proposed(fake_images.detach())

        d_loss_adv = F.relu(1.+out_adv).mean()
        d_loss_cls = F.cross_entropy(out_cls, c_)
        d_loss_fake = (d_loss_adv + d_loss_cls)

        d_loss = d_loss_real + d_loss_fake

        reset_grad()
        d_loss.backward()
        proposed_optimizer.step()

        log['D_fake_adv'] = d_loss_adv.item()
        log['D_fake_cls'] = d_loss_cls.item()
        log['D_fake_loss'] = d_loss_fake.item()

        log['D_loss'] = d_loss.item()


        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #
        # Compute loss with fake images
        z = torch.randn(batch, nz, 1, 1).to(device) # mean==0, std==1
        c_ = (torch.rand(batch, 1) * ncls).long().squeeze().to(device)
        cls = label2Glabel[c_].to(device)
        fake_images = generator(z, cls)

        out_cls, out_adv = proposed(fake_images)
        g_loss_adv = -out_adv.mean()
        g_loss_cls = F.cross_entropy(out_cls, c_)

        g_loss = (g_loss_adv + g_loss_cls)

        reset_grad()
        g_loss.backward()
        generator_optimizer.step()

        log['G_adv'] = g_loss_adv.item()
        log['G_cls'] = g_loss_cls.item()
        log['G_loss'] = g_loss.item()

        if i == 90:
            print(f'Epoch [{epoch}/{num_epochs}], Step [{i}/{len(train_data_loader)}]')
            for l_name, l_value in log.items():
                print(f'{l_name}: {l_value:.4}')


    for test_idx, data in enumerate(test_data_loader):
        img, target = data
        img, target = img.to(device), target.to(device)
        batch = img.size(0)

        model.eval()
        test_pred = model(img)
        # loss = F.cross_entropy(pred, target)
        # test_loss += loss

        test_pred = test_pred.argmax(-1)
        test_accuracy += (test_pred == target).sum()/batch

    test_accuracy = test_accuracy/len(test_data_loader)

    if test_best_accuracy < test_accuracy:
        test_best_accuracy = test_accuracy
        test_best_accuracy_epoch = epoch

    print(f"test_best_acc: {test_best_accuracy:.4}({test_best_accuracy_epoch})")

