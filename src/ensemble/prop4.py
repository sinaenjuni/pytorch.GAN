import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from utiles.dataset import CIFAR10, MNIST

name = 'prop4/weighted_cDCGAN_test2'
tensorboard_path = f'../../tb_logs/{name}'

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

tb = getTensorboard(tensorboard_path)

# Hyper-parameters configuration
num_epochs = 200
batch_size = 64
learning_rate = 0.002

nc=3
ngf=32
ndf=32
ngpu=1


# Dataset modify
# dataset = MNIST(32)
dataset = CIFAR10()
train_dataset = dataset.getTrainDataset()
transformed_dataset, count = dataset.getTransformedDataset([0.5 ** i for i in range(len(dataset.classes))])
test_dataset = dataset.getTestDataset()

ce_weights = [1-(i/sum(count["transformed"])) for i in count["transformed"]]
ce_weights = torch.FloatTensor(ce_weights).to(device)
print(ce_weights)

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


# custom weights initialization called on netG and netD
# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            # nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.ReLU(inplace=True),
            # state size. (ndf*8) x 4 x 4

            # nn.Sigmoid()
        )

        self.classifier1 = nn.Conv2d(ndf * 4, ndf * 4, 4, 1, 0, bias=False)
        self.classifier2 = nn.Linear(ndf * 4, 10)

    def forward(self, input):
        # return self.main(input)
        out = self.main(input)
        out = self.classifier1(out)
        out = F.avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.classifier2(out)
        return out


# Device setting
model = Discriminator(ngpu).to(device)

# SAVE_PATH = f'../../weights/cDCGAN/cifar10_lt_test1/'
# model.load_state_dict(torch.load(SAVE_PATH + 'D_200.pth'), strict=False)

criterion = torch.nn.CrossEntropyLoss(weight=ce_weights).to(device)  # 비용 함수에 소프트맥스 함수 포함되어져 있음.
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
    class_names = [i for i in count['class']]
    df_cm = pd.DataFrame(arr, class_names, class_names)

    fig = plt.figure(figsize=(9, 6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
    plt.xlabel("prediction")
    plt.ylabel("label (ground truth)")
    plt.tight_layout()
    tb.add_figure(tag='confusion_matrix', global_step=epoch + 1, figure=fig)
    # plt.close(fig)

    # if best_loss > loss_test:
    #     save_path = f'../../weights/{name}/{epoch+1}_{loss_test}.pth'
    #     if not os.path.exists(os.path.split(save_path)[0]):
    #         os.makedirs(os.path.split(save_path)[0])
    #     torch.save(model.state_dict(), save_path)
    #     best_loss = loss_test