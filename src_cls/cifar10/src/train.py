import torch
import torch.nn as nn
from torch.utils.data import Subset, ConcatDataset
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import os

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sns
import numpy as np

import sys
sys.path.append('/'.join(os.path.dirname(__file__).split('/')[:-2]))
from utils.dataset import sliceDataset


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# Hyperparameters
image_size = 32
batch_size = 100
num_epochs = 200
learning_rate = 0.001

name = 'cifar10/transformed'

# Weights save path
weights_path = '../weights/' + name
if not os.path.exists(weights_path):
    os.makedirs(weights_path)

# TensorBoard define
log_dir = '../../../tb_logs/cls/' + name
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
tb = SummaryWriter(log_dir=log_dir)

# Dataset define
transform = transforms.Compose([
    # transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5],  # 3 for greyscale channels
                         std=[0.5,0.5,0.5])])

# MNIST dataset
train_dataset = torchvision.datasets.CIFAR10(root='../../../data/',
                                   train=True,
                                   transform=transform,
                                   download=True)

test_dataset = torchvision.datasets.CIFAR10(root='../../../data/',
                                   train=False,
                                   transform=transform,
                                   download=True)

# Dataset modify
classes = train_dataset.class_to_idx
train_labels = torch.tensor(train_dataset.targets)
ratio = [0.5**i for i in range(len(classes))]
print(classes)
print(train_labels)
print(ratio)

transformed_dataset, count= sliceDataset(dataset=train_dataset,
                                     class_index=classes,
                                     labels=train_labels,
                                     lratio=ratio)

fig = plt.figure(figsize=(9, 6))
sns.barplot(
    data=count,
    x="class",
    y="original"
)
plt.tight_layout()
tb.add_figure(tag='original_data_dist', figure=fig)
# plt.show()

fig = plt.figure(figsize=(9, 6))
sns.barplot(
    data=count,
    x="class",
    y="transformed"
)
plt.tight_layout()
tb.add_figure(tag='transformed_data_dist', figure=fig)
# plt.show()




# Data loader
train_data_loader = torch.utils.data.DataLoader(dataset=transformed_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)

# 28 -> 14 -> 7 -> 3
class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.keep_prob = 0.5
        # L1 ImgIn shape=(?, 32, 32, 3)
        #    Conv     -> (?, 32, 32, 32)
        #    Pool     -> (?, 16, 16, 32)
        self.layer1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # L2 ImgIn shape=(?, 16, 16, 32)
        #    Conv      ->(?, 16, 16, 64)
        #    Pool      ->(?, 8, 8, 64)
        self.layer2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # L3 ImgIn shape=(?, 8, 8, 64)
        #    Conv      ->(?, 8, 8, 128)
        #    Pool      ->(?, 4, 4, 128)
        self.layer3 = torch.nn.Sequential(
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2))

        # L4 FC 4x4x128 inputs -> 625 outputs
        self.fc1 = torch.nn.Linear(4 * 4 * 128, 625, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU(),
            torch.nn.Dropout(p = 1 - self.keep_prob))

        # L5 Final FC 625 inputs -> 10 outputs
        self.fc2 = torch.nn.Linear(625, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.view(out.size(0), -1)   # Flatten them for FC
        out = self.layer4(out)
        out = self.fc2(out)
        return out


model = CNN().to(device)
print(model)
criterion = torch.nn.CrossEntropyLoss().to(device)  # 비용 함수에 소프트맥스 함수 포함되어져 있음.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Start training
total_step = len(train_data_loader)
for epoch in range(num_epochs):
    epoch_loss = 0
    epoch_acc = 0
    for i, (images, labels) in enumerate(train_data_loader):
        images = images.to(device)
        labels = labels.to(device)

        pred = model(images).to(device)
        optimizer.zero_grad()
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()

        pred = pred.argmax(-1)
        acc_train = torch.sum(pred == labels)
        acc_train = acc_train / labels.size(0)

        # if (i + 1) % 10 == 0:
        print('Epoch [{}/{}], Step [{}/{}], loss:{:.4f}, acc:{:.4f}'
              .format(epoch+1, num_epochs,
                      i + 1, total_step,
                      loss.item(), acc_train.item()))

    with torch.no_grad():
        arr = torch.zeros((len(classes), len(classes)), dtype=torch.long)

        for i, (images, labels) in enumerate(test_data_loader):
            images = images.to(device)
            labels = labels.to(device)

            pred = model(images).to(device)

            pred = pred.argmax(-1)
            acc_test = torch.sum(pred == labels)
            acc_test = acc_test / labels.size(0)

            arr += confusion_matrix(labels.cpu(), pred.cpu())

        tb.add_scalar(global_step=epoch+1, tag='loss', scalar_value=loss.item())
        tb.add_scalars(global_step=epoch+1,
                       main_tag='acc',
                       tag_scalar_dict={'train':acc_train,
                                        'test': acc_test})

        class_names = [f'{i}' for i in range(10)]
        df_cm = pd.DataFrame(arr.cpu().numpy(), class_names, class_names)
        fig = plt.figure(figsize=(9, 6))
        sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
        plt.xlabel("prediction")
        plt.ylabel("label (ground truth)")
        plt.tight_layout()
        tb.add_figure(tag='confusion_matrix', global_step=epoch + 1, figure=fig)
        # plt.close(fig)
