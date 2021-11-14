import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
import os

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# Hyperparameters
image_size = 28
batch_size = 1

# TensorBoard define
log_dir = '../tb_logs/cls/'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
tb = SummaryWriter(log_dir=log_dir)


transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],  # 1 for greyscale channels
                         std=[0.5])])

# MNIST dataset
mnist = torchvision.datasets.MNIST(root='../data/',
                                   train=True,
                                   # transform=transform,
                                   download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=batch_size,
                                          shuffle=True)


class model(nn.Module):
    def __init__(self):

        layer = nn.Sequential(nn)
