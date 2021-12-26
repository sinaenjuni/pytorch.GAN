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
ngf = 32

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
            nn.ReLU(True),
            nn.Conv2d(in_channels=ndf,
                      out_channels=ndf*2,
                      kernel_size=3,
                      stride=2,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(ndf*2),
            nn.ReLU(True),
            nn.Conv2d(in_channels=ndf*2,
                      out_channels=ndf*4,
                      kernel_size=3,
                      stride=2,
                      padding=0,
                      bias=False),
            nn.BatchNorm2d(ndf*4),
            nn.ReLU(True),
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
            print(idx)
        print(discriminator_input.size())

        discriminator = self.discriminator(discriminator_input)
        discriminator = discriminator.squeeze()

        return classifier, discriminator


input_tensor = torch.rand((32,3,32,32)).to(device)
# Define model
model = resnet32(num_classes=10, use_norm=True).to(device)
proposed = Propose(model).to(device)
generator = Generator(nz, nc, ncls, ngf)

classifier, discriminator = proposed(input_tensor)
print(classifier.size(), discriminator.size())


proposed_optimizer = torch.optim.Adam(proposed.parameters(), lr=learning_rate, betas=(beta1, beta2))
generator_optimizer = torch.optim.Adam(generator.parameters(), lr=learning_rate, betas=(beta1, beta2))


# x = input_tensor
# for idx, layer in enumerate(model.children()):
#     x = layer(x)
#     print(idx)
#     print(x.size())
#     if idx == 3:
#         break


# print(model)
# summary(model, torch.rand((32,3,32,32)).to(device))

# propose = Propose(model)
# print(propose)
# print(propose(input_tensor))

# shared_model = model.(input_tensor)
# print(shared_model)
# summary(shared_model, torch.rand((64,3,32,32)).to(device))

# classifier = nn.Sequential(*list(model.children())[4:])
# print(classifier)
# summary(classifier, torch.rand((64,32,16,16)).to(device))


# summary(propose, torch.rand((32,3,32,32)).to(device))
# G = Generator(100, 3, 10, 128)



# summary(model, torch.rand((32,3,32,32)).to(device))

# model = nn.Sequential(*list(model.children())[:4])
# print(list(model.modules()))
# last_module = model.out_channels
# print(last_module)
# summary(model, torch.rand((32,3,32,32)).to(device))

# print(model)
# for i, module in enumerate(model.modules()):
#     print(i)
#     print(module)

# middle_feature = model.children()
# for i, module in enumerate(middle_feature):
#     print(i, module)
# print(list(middle_feature))
# summary(model, torch.rand((64,3,32,32)).to(device))

# D =

# print(model)





# # Define optimizer
# optimizer = torch.optim.SGD(model.parameters(),
#                             lr=learning_rate,
#                             momentum=momentum,
#                             weight_decay=weight_decay)
#
# train_best_accuracy = 0
# train_best_accuracy_epoch = 0
# test_best_accuracy = 0
# test_best_accuracy_epoch = 0
#
# # Training model
# for epoch in range(num_epochs):
#     train_accuracy = 0
#     test_accuracy = 0
#     train_loss = 0
#     test_loss = 0
#     for train_idx, data in enumerate(train_data_loader):
#         img, target = data
#         img, target = img.to(device), target.to(device)
#         batch = img.size(0)
#
#         model.train()
#         pred = model(img)
#
#         loss = F.cross_entropy(pred, target)
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()
#
#         train_loss += loss
#         pred = pred.argmax(-1)
#         train_accuracy += (pred == target).sum()/batch
#         # print(f"epochs: {epoch}, iter: {train_idx}/{len(train_data_loader)}, loss: {loss.item()}")
#
#
#     for test_idx, data in enumerate(test_data_loader):
#         img, target = data
#         img, target = img.to(device), target.to(device)
#         batch = img.size(0)
#
#         model.eval()
#         pred = model(img)
#         # loss = F.cross_entropy(pred, target)
#         # test_loss += loss
#
#         pred = pred.argmax(-1)
#         test_accuracy += (pred == target).sum()/batch
#
#
#     train_loss = train_loss/len(train_data_loader)
#     train_accuracy = train_accuracy/len(train_data_loader)
#     test_accuracy = test_accuracy/len(test_data_loader)
#
#     if train_best_accuracy < train_accuracy:
#         train_best_accuracy = train_accuracy
#         train_best_accuracy_epoch = epoch
#     if test_best_accuracy < test_accuracy:
#         test_best_accuracy = test_accuracy
#         test_best_accuracy_epoch = epoch
#
#     print(f"epochs: {epoch}, "
#           f"train_loss: {train_loss:.4}, "
#           # f"test_loss: {test_loss/len(test_data_loader):.4}. "
#           f"train_acc: {train_accuracy:.4}, "
#           f"test_acc: {test_accuracy:.4}, "
#           f"train_best_acc: {train_best_accuracy:.4}({train_best_accuracy_epoch}), "
#           f"test_best_acc: {test_best_accuracy:.4}({test_best_accuracy_epoch})")
















