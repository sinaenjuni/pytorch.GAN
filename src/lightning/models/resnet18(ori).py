import pytorch_lightning as pl
from sklearn.metrics import confusion_matrix


class Restnet_classifier(pl.LightningModule):
    def __init__(self):
        self.model = resnet18(num_classes=10).to(device)
        self.model.fc = nn.Linear(in_features=512, out_features=10).to(device)

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        image, label = batch
        logit = self(image)
        loss = self.criterion(logit, label)
        pred = logit.argmax(-1)

        train_cm = confusion_matrix(pred, label)
        train_acc = train_cm.trace() / train_cm.sum()
        train_acc_per_cls = train_cm.diagonal() / train_cm.sum(0)

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict({"train_acc":train_acc,
                       "train_acc_per_cls": " ".join([f"({idx}) {acc:.4}" for idx, acc in enumerate(train_acc_per_cls)])})
        return loss

    def validation_step(self, batch, batch_idx):
        image, label = batch
        logit = self(image)
        loss = self.criterion(label, logit)
        pred = logit.argmax(-1)

        cm = confusion_matrix(pred, label)
        val_acc = cm.trace() / cm.sum()
        val_acc_per_cls = cm.diagonal() / cm.sum(0)

        metrics = {"val_loss":loss,
                   "val_acc": val_acc,
                   "val_acc_per_cls": " ".join([f"({idx}) {acc:.4}" for idx, acc in enumerate(val_acc_per_cls)])}
        self.log_dict(metrics)
        return metrics

    # def validation_step_end(self, val_step_outputs):
    #     val_acc = val_step_outputs['val_acc'].cpu()
    #     val_loss = val_step_outputs['val_loss'].cpu()
    #
    #     self.log('validation_acc', val_acc, prog_bar=True)
    #     self.log('validation_loss', val_loss, prog_bar=True)

    def test_step(self, batch, batch_idx):
        image, label = batch
        label_logits = self(image)
        loss = self.criterion(label_logits, label.long())

        cm = confusion_matrix(pred, label)
        val_acc = cm.trace() / cm.sum()
        val_acc_per_cls = cm.diagonal() / cm.sum(0)

        metrics = {"test_loss":loss,
                   "test_acc": val_acc,
                   "test_acc_per_cls": " ".join([f"({idx}) {acc:.4}" for idx, acc in enumerate(val_acc_per_cls)])}
        self.log_dict(metrics)
        return metrics

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        optimizer = torch.optim.SGD(self.parameters(),
                                    momentum=self.hparams.momentum,
                                    lr=self.hparams.learning_rate,
                                    weight_decay=self.hparams.weight_dacay,
                                    nesterov=nesterov)

        return optimizer

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = parent_parser.add_argument_group("MLP_MNIST_Classifier")
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        parser.add_argument('--momentum', type=float, default=0.0001)
        parser.add_argument('--weight_decay', type=float, default=0.0001)
        parser.add_argument('--weight_decay', type=float, default=0.0001)
        return parent_parser

import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utiles.tensorboard import getTensorboard
from utiles.data import getSubDataset
from utiles.imbalance_cifar10_loader import ImbalanceCIFAR10DataLoader
# from models.resnet_s import resnet32
# from models.resnet import resnet18
from torchvision.models import resnet18
from models.resnet import resnet34


# Define hyper-parameters
# name = "pytorch.GAN/experiment2/resnet_s/cifar10_0.01_sampler_LSGAN/"
name = "pytorch.GAN/experiment2/resnet18(ori)/cifar10_0.1"
tensorboard_path = f"/home/sin/tb_logs/{name}"
logging_path = f"/home/sin/logging/{name}"
weight_path = f"/home/sin/weights/{name}"

if not os.path.exists(tensorboard_path):
    os.makedirs(tensorboard_path)
if not os.path.exists(logging_path):
    os.makedirs(logging_path)
if not os.path.exists(weight_path):
    os.makedirs(weight_path)


import logging
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(level=logging.DEBUG)

# target_weight_path = f'/home/sin/weights/pytorch.GAN/experiment2/gan/cifar10_0.01_sampler_LSGAN/'


num_workers = 4
num_epochs = 200
batch_size = 128
imb_factor = 0.1

learning_rate = 0.1
weight_decay = 5e-4
momentum = 0.9
nesterov = True
# target_epoch = [20, 40, 60, 80, 100]
step1 = 160
step2 = 180
gamma = 0.1
warmup_epoch = 5


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)


# Define Tensorboard
tb = getTensorboard(tensorboard_path)

# Define DataLoader
train_data_loader = ImbalanceCIFAR10DataLoader(data_dir='~/data',
                                              batch_size=batch_size,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              training=True,
                                              imb_factor=imb_factor)

test_data_loader = ImbalanceCIFAR10DataLoader(data_dir='~/data',
                                              batch_size=batch_size,
                                              shuffle=False,
                                              num_workers=num_workers,
                                              training=False)


print("Number of train dataset", len(train_data_loader.dataset))
print("Number of test dataset", len(test_data_loader.dataset))

cls_num_list = train_data_loader.cls_num_list
cls2idx = list(train_data_loader.dataset.class_to_idx.values())
print(cls_num_list)
print(cls2idx)


# Define optimizer
# optimizer = torch.optim.Adam(model.parameters(),
#                             lr=learning_rate,
#                             # momentum=momentum,
#                             weight_decay=weight_decay)

def lr_lambda(epoch):
    if epoch >= step2:
        lr = gamma * gamma
    elif epoch >= step1:
        lr = gamma
    else:
        lr = 1

    """Warmup"""
    if epoch < warmup_epoch:
        lr = lr * float(1 + epoch) / warmup_epoch
    print(lr)
    return lr

# for target_idx in range(len(target_epoch)):
#     if target_idx != 0:
#         logger.removeHandler(logging.FileHandler(logging_path + f'./{target_epoch[target_idx - 1]}_logging.txt'))
#     logger.addHandler(logging.FileHandler(logging_path + f'./{target_epoch[target_idx]}_logging.txt'))


    # Define model


    # model.load_state_dict(torch.load(target_weight_path + f"D_{target_epoch[target_idx]}.pth"), strict=False)


lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

train_best_loss = 0.0
train_best_acc = 0
train_best_acc_epoch = 0

test_best_loss = 0.0
test_best_acc = 0
test_best_acc_epoch = 0
test_best_acc_per_cls = []

# Training model
for epoch in range(num_epochs):
    train_loss = 0.0
    test_loss = 0.0

    train_pred = np.array([])
    train_label = np.array([])
    test_pred = np.array([])
    test_label = np.array([])

    for train_idx, (image, label) in enumerate(train_data_loader):
        image, label = image.to(device), label.to(device)
        batch = image.size(0)
        optimizer.zero_grad()

        model.train()
        pred = model(image)

        loss = F.cross_entropy(pred, label)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        train_pred = np.append(train_pred, pred.argmax(-1).tolist())
        train_label = np.append(train_label, label.tolist())

    model.eval()
    with torch.no_grad():
        for test_idx, (image, label) in enumerate(test_data_loader):
            image, label = image.to(device), label.to(device)
            batch = image.size(0)

            pred = model(image)
            loss = F.cross_entropy(pred, label)

            test_loss += loss.item()
            test_pred = np.append(test_pred, pred.argmax(-1).tolist())
            test_label = np.append(test_label, label.tolist())


    train_loss = train_loss / len(train_data_loader)
    train_cm = confusion_matrix(train_pred, train_label, labels=cls2idx)
    train_acc = train_cm.trace()/train_cm.sum()
    train_acc_per_cls = train_cm.diagonal() / train_cm.sum(0)


    test_loss = test_loss / len(test_data_loader)
    test_cm = confusion_matrix(test_pred, test_label, labels=cls2idx)
    test_acc = test_cm.trace()/test_cm.sum()
    test_acc_per_cls = test_cm.diagonal() / test_cm.sum(0)


    if train_best_acc < train_acc:
        train_best_loss = train_loss
        train_best_acc = train_acc
        train_best_acc_epoch = epoch

    if test_best_acc < test_acc:
        test_best_loss = test_loss
        test_best_acc = test_acc
        test_best_acc_epoch = epoch
        test_best_acc_per_cls = test_acc_per_cls


    logger.info(f"Epoch: {epoch}/{num_epochs}")
    logger.info(f"(Train)")
    logger.info(f"loss: {train_loss:>7.4}")
    logger.info(f"acc: {train_acc:>7.4}")
    logger.info(" ".join([f"({idx}) {acc:>7.4}" for idx, acc in enumerate(train_acc_per_cls)]))

    logger.info(f"(Test)")
    logger.info(f"loss: {test_loss:>7.4}")
    logger.info(f"acc: {test_acc:>7.4}")
    logger.info(" ".join([f"({idx}) {acc:>7.4}" for idx, acc in enumerate(test_acc_per_cls)]))

    logger.info(f"(Best)")
    logger.info(f"Epoch: {test_best_acc_epoch}")
    logger.info(f"loss: {test_best_loss:>7.4}")
    logger.info(f"acc: {test_best_acc:>7.4}")
    logger.info(" ".join([f"({idx}) {acc:>7.4}" for idx, acc in enumerate(test_acc_per_cls)]))


    print(max([param_group['lr'] for param_group in optimizer.param_groups]),
                min([param_group['lr'] for param_group in optimizer.param_groups]))
    lr_scheduler.step()








