import os
import torch
import torch.nn as nn
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

tensorboard_path = '../../tb_logs/ResNet18/test1'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

tb = getTensorboard(tensorboard_path)

# Hyper-parameters configuration
num_epochs = 200
batch_size = 32
learning_rate = 0.0002


# Transformation define
transform = transforms.Compose([
    # transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],  # 3 for greyscale channels
                         std=[0.5, 0.5, 0.5])])

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


# Model define
model = ResNet18().to(device)
print(model)


criterion = torch.nn.CrossEntropyLoss().to(device)  # 비용 함수에 소프트맥스 함수 포함되어져 있음.
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# Start training
total_step = len(train_data_loader)
for epoch in range(num_epochs):
    loss_train = 0
    acc_train = 0

    for i, (images, labels) in enumerate(train_data_loader):
        images = images.to(device)
        labels = labels.to(device)

        pred = model(images).to(device)
        optimizer.zero_grad()
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()


        pred = pred.argmax(-1)
        acc = torch.sum(pred == labels).float()
        acc = acc / labels.size(0)

        loss_train += loss.item()
        acc_train += acc.item()

        if (i + 1) % 100 == 0:
            # print('Epoch [{}/{}], Step [{}/{}], loss:{:.4f}, acc:{:.4f}'
            #       .format(epoch+1, num_epochs,
            #               i + 1, total_step,
            #               loss.item(), acc_train.item()))
            print('Epoch [{}/{}], Step [{}/{}], loss:{:.4f}, acc:{:.4f}'
                  .format(epoch + 1, num_epochs,
                          i + 1, total_step,
                          loss_train/i, acc_train/i))


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
                       tag_scalar_dict={'train': acc_train,
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
