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

from utiles.tensorboard import getTensorboard
from utiles.data import getSubDataset
from utiles.imbalance_cifar10_loader import ImbalanceCIFAR10DataLoader
from models.resnet_s import resnet32


# Define hyper-parameters
name = 'cifar_cls'
tensorboard_path = f'../../tb_logs/ResNet_s/{name}'

num_workers = 4
num_epochs = 200
batch_size = 128

learning_rate = 0.1
weight_decay = 5e-4
momentum = 0.9
nesterov = True


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# Define Tensorboard
tb = getTensorboard(tensorboard_path)


# Define dataset
normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                 std=[0.2023, 0.1994, 0.2010])
train_transform = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    normalize,
])
test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                   train=True,
                                   transform=train_transform,
                                   download=True)

test_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                   train=False,
                                   transform=test_transform,
                                   download=True)


# Define DataLoader
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=num_workers)

train_dataset_im = ImbalanceCIFAR10DataLoader(data_dir='../../data',
                                              batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers, training=True, imb_factor=0.01)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=batch_size,
                         shffle=False,
                         num_workers=num_workers)


# Define model
model = resnet32(num_classes=10, use_norm=True).to(device)
print(model)


# Define optimizer
optimizer = torch.optim.SGD(model.parameters(),
                            lr=learning_rate,
                            momentum=momentum,
                            weight_decay=weight_decay)



for epoch in num_epochs:
    for batch_idx, data in enumerate(train_loader):
        img, target = data.to(device)

        pred = model(img)

        loss = F.cross_entropy(pred, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f"epochs: {epoch}, loss: {loss.item()}")
















