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
from utiles.imbalance_cifar10_loader import ImbalanceCIFAR10DataLoader

name = 'experiments/classifier/cifar10_dist/test1(200)'
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

classes = ["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

# Dataset define
train_data_loader = ImbalanceCIFAR10DataLoader(data_dir='../../data',
                                               batch_size=batch_size,
                                               shuffle=True,
                                               training=True,
                                               imb_factor=0.01)

test_data_loader = ImbalanceCIFAR10DataLoader(data_dir='../../data',
                                              batch_size=batch_size,
                                              shuffle=False,
                                              training=False,
                                              imb_factor=0.01)
print(len(train_data_loader.dataset))
print(len(test_data_loader.dataset))

class_to_index = train_data_loader.dataset.class_to_idx
num_per_cls_dict = train_data_loader.dataset.num_per_cls_dict
class_name_and_counts = {name[0]: counts[1] for name, counts in zip(class_to_index.items(),
                                                                    num_per_cls_dict.items())}
print(class_to_index)
print(num_per_cls_dict)
print(class_name_and_counts)

num_per_cls_dict = pd.DataFrame(list(class_name_and_counts.items()),
                                columns=['name', 'counts'])

print(num_per_cls_dict)

fig = plt.figure(figsize=(9, 6))
sns.barplot(
    data=num_per_cls_dict,
    x="name",
    y="counts"
)
plt.tight_layout()
# tb.add_figure(tag='original_data_dist', figure=fig)
plt.show()



# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


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
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
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

SAVE_PATH = f'../../weights/experiments/DCGAN/cifar10_dist/test1/D_200.pth'
model.load_state_dict(torch.load(SAVE_PATH), strict=False)

criterion = torch.nn.CrossEntropyLoss().to(device)  # 비용 함수에 소프트맥스 함수 포함되어져 있음.
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate,
                      momentum=0.9, weight_decay=5e-4)


# Start training
num_train_step = len(train_data_loader)
num_test_step = len(test_data_loader)

num_train_dataset = len(train_data_loader.dataset)
num_test_dataset = len(test_data_loader.dataset)

log = {}
log["epoch"] = 0
log["loss_train"] = 0
log["loss_test"] = 0
log["acc_train"] = 0
log["acc_test"] = 0
log["best_loss_train"] = float("inf")
log["best_loss_train_epoch"] = 0
log["best_loss_test"] = float("inf")
log["best_loss_test_epoch"] = 0
log["best_acc_train"] = 0
log["best_acc_train_epoch"] = 0
log["best_acc_test"] = 0
log["best_acc_test_epoch"] = 0




for epoch in range(num_epochs):
    labels_train = np.array([])
    preds_train = np.array([])
    loss_train = np.array([])

    log["epoch"] = epoch + 1

    for i, (images, labels) in enumerate(train_data_loader):

        model.train()
        images = images.to(device)
        labels = labels.to(device)

        preds = model(images).to(device)

        optimizer.zero_grad()
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        preds = preds.argmax(-1)

        labels_train = np.append(labels_train, labels.cpu().numpy())
        preds_train = np.append(preds_train, preds.cpu().numpy())
        loss_train = np.append(loss_train, loss.item())

    with torch.no_grad():
        model.eval()
        labels_test = np.array([])
        preds_test = np.array([])
        loss_test = np.array([])

        for i, (images, labels) in enumerate(test_data_loader):
            images = images.to(device)
            labels = labels.to(device)

            preds = model(images).to(device)
            loss = criterion(preds, labels)

            preds = preds.argmax(-1)

            loss_test = np.append(loss_test, loss.item())
            labels_test = np.append(labels_test, labels.cpu().numpy())
            preds_test = np.append(preds_test, preds.cpu().numpy())

    log["loss_train"] = loss_train.mean()
    log["acc_train"] = (labels_train == preds_train).mean()

    log["loss_test"] = loss_test.mean()
    log["acc_test"] = (labels_test == preds_test).mean()

    if log["best_loss_train"] > log["loss_train"]:
        log["best_loss_train"] = log["loss_train"]
        log["best_loss_train_epoch"] = log["epoch"]
    if log["best_loss_test"] > log["loss_test"]:
        log["best_loss_test"] = log["loss_test"]
        log["best_loss_test_epoch"] = log["epoch"]

    if log["best_acc_train"] < log["acc_train"]:
        log["best_acc_train"] = log["acc_train"]
        log["best_acc_train_epoch"] = log["epoch"]
    if log["best_acc_test"] < log["acc_test"]:
        log["best_acc_test"] = log["acc_test"]
        log["best_acc_test_epoch"] = log["epoch"]

            # print(f'Epoch:       {epoch + 1}/{num_epochs}     \n'
            #       f'Step:        {i + 1}/{num_train_dataset}  \n'
            #       f'loss_train:  {loss_train.mean():.4f}      \n'
            #       f'acc_train:   {acc_train.mean():.4f}       \n')

    unique, counts = np.unique(labels_test, return_counts=True)
    match = (labels_test == preds_test)

    counts_per_class = {str(unique): f"{match[labels_test == unique].sum()}/{counts}" for unique, counts in zip(unique, counts)}
    acc_per_class =    {str(unique): match[labels_test == unique].sum() / counts for unique, counts in zip(unique, counts)}
    acc = match.mean()


    print("==================================================")
    for key, val in log.items():
        if any(keyword in key for keyword in ['epoch', 'step']):
            print(f"{key:<30}:{val:>10}")
        else:
            print(f"{key:<30}:{val:>10.4f}")

        # print(counts_per_class)
        # print(acc_per_class)
        # print(acc)
        # print(confusion_matrix(labels_test, preds_test))

            # print(labels, pred)
            # arr += confusion_matrix(labels.cpu(), pred.cpu())
        # print(labels_test)
        # print(arr)
        # print(acc_test)

    # loss_train /= num_train_step
    # loss_test /= num_test_step
    # acc_train /= num_train
    # acc_test /= num_test
    #
    tb.add_scalars(global_step=epoch+1,
                   main_tag='loss',
                   tag_scalar_dict={'train': loss_train.mean(),
                                     'test': loss_test.mean()})

    tb.add_scalars(global_step=epoch+1,
                   main_tag='acc',
                   tag_scalar_dict={'train': (labels_train == preds_train).mean(),
                                     'test': (labels_test == preds_test).mean()})

    # tb.add_text(global_step=epoch+1,
    #                tag='counts_per_class',
    #                text_string=str(counts_per_class))
    #
    # tb.add_text(global_step=epoch + 1,
    #             tag='acc_per_class',
    #             text_string=str(acc_per_class))

    arr = confusion_matrix(labels_test, preds_test)
    class_names = [i for i in classes]
    df_cm = pd.DataFrame(arr, class_names, class_names)

    fig = plt.figure(figsize=(9, 6))
    sns.heatmap(df_cm, annot=True, fmt="d", cmap='BuGn')
    plt.xlabel("prediction")
    plt.ylabel("label (ground truth)")
    plt.tight_layout()
    tb.add_figure(tag='confusion_matrix', global_step=epoch + 1, figure=fig)
    # plt.close(fig)
    #
    # if best_loss > loss_test:
    #     save_path = f'../../weights/{name}/{epoch+1}_{loss_test}.pth'
    #     if not os.path.exists(os.path.split(save_path)[0]):
    #         os.makedirs(os.path.split(save_path)[0])
    #     torch.save(model.state_dict(), save_path)
    #     best_loss = loss_test