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
                                              shuffle=True,
                                              num_workers=num_workers,
                                              training=True,
                                              imb_factor=0.01)

test_data_loader = ImbalanceCIFAR10DataLoader(data_dir='../../data',
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              training=False)


print("Number of train dataset", len(train_data_loader.dataset))
print("Number of test dataset", len(test_data_loader.dataset))

# Define model
model = resnet32(num_classes=10, use_norm=True).to(device)
print(model)


# Define optimizer
optimizer = torch.optim.Adam(model.parameters(),
                            lr=learning_rate,
                            # momentum=momentum,
                            weight_decay=weight_decay)

train_best_accuracy = 0
train_best_accuracy_epoch = 0
test_best_accuracy = 0
test_best_accuracy_epoch = 0

# Training model
for epoch in range(num_epochs):
    train_accuracy = 0
    test_accuracy = 0
    train_loss = 0
    test_loss = 0
    for train_idx, data in enumerate(train_data_loader):
        img, target = data
        img, target = img.to(device), target.to(device)
        batch = img.size(0)

        model.train()
        pred = model(img)

        loss = F.cross_entropy(pred, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_loss += loss
        pred = pred.argmax(-1)
        train_accuracy += (pred == target).sum()/batch
        # print(f"epochs: {epoch}, iter: {train_idx}/{len(train_data_loader)}, loss: {loss.item()}")


    for test_idx, data in enumerate(test_data_loader):
        img, target = data
        img, target = img.to(device), target.to(device)
        batch = img.size(0)

        model.eval()
        pred = model(img)
        # loss = F.cross_entropy(pred, target)
        # test_loss += loss

        pred = pred.argmax(-1)
        test_accuracy += (pred == target).sum()/batch


    train_loss = train_loss/len(train_data_loader)
    train_accuracy = train_accuracy/len(train_data_loader)
    test_accuracy = test_accuracy/len(test_data_loader)

    if train_best_accuracy < train_accuracy:
        train_best_accuracy = train_accuracy
        train_best_accuracy_epoch = epoch
    if test_best_accuracy < test_accuracy:
        test_best_accuracy = test_accuracy
        test_best_accuracy_epoch = epoch

    print(f"epochs: {epoch}, "
          f"train_loss: {train_loss:.4}, "
          # f"test_loss: {test_loss/len(test_data_loader):.4}. "
          f"train_acc: {train_accuracy:.4}, "
          f"test_acc: {test_accuracy:.4}, "
          f"train_best_acc: {train_best_accuracy:.4}({train_best_accuracy_epoch}), "
          f"test_best_acc: {test_best_accuracy:.4}({test_best_accuracy_epoch})")
















