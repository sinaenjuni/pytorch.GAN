import os

import numpy as np
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
batch_size = 64
learning_rate = 0.002


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
train_data_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

test_data_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)


# Model define
model = ResNet18().to(device)
print(model)


criterion = torch.nn.CrossEntropyLoss().to(device)  # 비용 함수에 소프트맥스 함수 포함되어져 있음.
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=0.9, weight_decay=5e-4)


# Start training
num_train_step = len(train_data_loader)
num_train = len(train_data_loader.dataset)

num_test_step = len(test_data_loader)
num_test = len(test_data_loader.dataset)
best_loss = float("inf")

for epoch in range(num_epochs):
    loss_train = 0
    acc_train = 0
    total_train = 0

    for i, (images, labels) in enumerate(train_data_loader):
        model.train()
        images = images.to(device)
        labels = labels.to(device)

        pred = model(images).to(device)
        optimizer.zero_grad()
        loss = criterion(pred, labels)
        loss.backward()
        optimizer.step()

        loss_train += loss.item()

        pred = pred.argmax(-1)
        acc_train += (pred == labels).sum().item()
        total_train += labels.size(0)

        # acc_train += acc.item()

        # if (i + 1) % 100 == 0:
            # print('Epoch [{}/{}], Step [{}/{}], loss:{:.4f}, acc:{:.4f}'
            #       .format(epoch+1, num_epochs,
            #               i + 1, total_step,
            #               loss.item(), acc_train.item()))
    print('Epoch [{}/{}], Step [{}/{}], loss:{:.4f}, acc:{:.4f}'
          .format(epoch + 1, num_epochs,
                  i + 1, num_train_step,
                  loss_train/num_train_step,
                  100.*acc_train/num_train))

    with torch.no_grad():
        model.eval()
        loss_test = 0
        acc_test = 0

        labels_test = []
        preds_test = []

        for i, (images, labels) in enumerate(test_data_loader):
            images = images.to(device)
            labels = labels.to(device)

            pred = model(images).to(device)
            loss = criterion(pred, labels)
            loss_test += loss.item()

            pred = pred.argmax(-1)
            acc_test += (pred == labels).sum().item()

            labels_test += labels.tolist()
            preds_test += pred.tolist()


        # print(loss_test / num_test_step)
        # print(acc_test / num_test)
        # print(labels_test, preds_test)
        # print(confusion_matrix(labels_test, preds_test))
            # print(labels, pred)
            # arr += confusion_matrix(labels.cpu(), pred.cpu())
        # print(labels_test)
        # arr = confusion_matrix(labels_test, preds_test)
        # print(arr)
        # print(acc_test)

    loss_train /= num_train_step
    loss_test /= num_test_step
    acc_train /= num_train
    acc_test /= num_test

    tb.add_scalars(global_step=epoch+1,
                   main_tag='loss',
                   tag_scalar_dict={'train':loss_train,
                                    'test':loss_test})
    tb.add_scalars(global_step=epoch+1,
                   main_tag='acc',
                   tag_scalar_dict={'train': acc_train,
                                    'test': acc_test})

    arr = confusion_matrix(labels_test, preds_test)
    class_names = [i for i in classes.keys()]
    df_cm = pd.DataFrame(arr, class_names, class_names)

    fig = plt.figure(figsize=(9, 6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
    plt.xlabel("prediction")
    plt.ylabel("label (ground truth)")
    plt.tight_layout()
    tb.add_figure(tag='confusion_matrix', global_step=epoch + 1, figure=fig)
    # plt.close(fig)

    if best_loss > loss_test:
        save_path = f'../../weights/resnet18/test1_{loss_test}.pth'
        if not os.path.exists(os.path.split(save_path)[0]):
            os.makedirs(os.path.split(save_path)[0])
        torch.save(model.state_dict(), save_path)
        best_loss = loss_test
