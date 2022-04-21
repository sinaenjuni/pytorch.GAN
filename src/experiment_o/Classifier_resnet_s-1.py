import os

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix

from utiles.tensorboard import getTensorboard
from utiles.data import getSubDataset
from utiles.imbalance_cifar10_loader import ImbalanceCIFAR10DataLoader
from models.resnet_s import resnet32


# Define hyper-parameters
name = "pytorch.GAN/experiment2/resnet_s/cifar10_0.01_sampler_WGAN/"
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

target_weight_path = f'/home/sin/weights/pytorch.GAN/experiment2/gan/cifar10_0.01_sampler_WGAN/'


num_workers = 4
num_epochs = 400
batch_size = 128
imb_factor = 0.01

learning_rate = 0.1
weight_decay = 5e-4
momentum = 0.9
nesterov = True
target_epoch = [20, 40, 60, 80, 100]
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

for target_gen in target_epoch:
    logger.addHandler(logging.FileHandler(logging_path + f'./{target_gen}_logging.txt'))

    # Define model
    model = resnet32(num_classes=10, use_norm=True).to(device)
    model.load_state_dict(torch.load(target_weight_path + f"D_{target_gen}.pth"), strict=False)

    optimizer = torch.optim.SGD(model.parameters(),
                                momentum=momentum,
                                lr=learning_rate,
                                weight_decay=weight_decay,
                                nesterov=nesterov)
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








