import os
import torch
import torchvision
import torch.nn as nn
from torchvision import transforms
from torchvision.utils import make_grid
from torch.utils.tensorboard import SummaryWriter
from utiles.imbalance_mnist_loader import ImbalanceMNISTDataLoader
import matplotlib.pyplot as plt

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# TensorBoard define
# log_dir = '../../tb_logs/vanillaGAN/test3'
# if not os.path.exists(log_dir):
#     os.makedirs(log_dir)
# tb = SummaryWriter(log_dir=log_dir)

# Hyper-parameters
image_size = 32
noise_dim = 100
hidden_size = 256
batch_size = 64

epochs = 200
learning_rate_g = 0.0002
learning_rate_d = 0.0002
beta1 = 0.5
beta2 = 0.999
sample_dir = '../samples'

fixed_noise = torch.randn(batch_size, noise_dim).to(device)

# Create a directory if not exists
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# Image processing
# transform = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=(0.5, 0.5, 0.5),   # 3 for RGB channels
#                                      std=(0.5, 0.5, 0.5))])

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],  # 1 for greyscale channels
                         std=[0.5])])

# MNIST dataset
mnist = torchvision.datasets.MNIST(root='../../data/',
                                   train=True,
                                   transform=transform,
                                   download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=batch_size,
                                          shuffle=True)


# def initialize_weights(net):
#     for m in net.modules():
#         if isinstance(m, nn.Conv2d):
#             m.weight.data.normal_(0, 0.02)
#             m.bias.data.zero_()
#         elif isinstance(m, nn.ConvTranspose2d):
#             m.weight.data.normal_(0, 0.02)
#             m.bias.data.zero_()
#         elif isinstance(m, nn.Linear):
#             m.weight.data.normal_(0, 0.02)
#             m.bias.data.zero_()
#
# class Generator(nn.Module):
#     # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
#     # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
#     def __init__(self, input_dim=100, output_dim=1, image_size=32):
#         super(Generator, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.image_size = image_size
#
#         self.fc = nn.Sequential(
#             nn.Linear(self.input_dim, 1024),
#             nn.BatchNorm1d(1024),
#             nn.ReLU(),
#             nn.Linear(1024, 128 * (self.image_size // 4) * (self.image_size // 4)),
#             nn.BatchNorm1d(128 * (self.image_size // 4) * (self.image_size // 4)),
#             nn.ReLU(),
#         )
#         self.deconv = nn.Sequential(
#             nn.ConvTranspose2d(128, 64, 4, 2, 1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.ConvTranspose2d(64, self.output_dim, 4, 2, 1),
#             nn.Tanh(),
#         )
#         initialize_weights(self)
#
#     def forward(self, input):
#         x = self.fc(input)
#         x = x.view(-1, 128, (self.image_size // 4), (self.image_size // 4))
#         x = self.deconv(x)
#
#         return x
#
# class Discriminator(nn.Module):
#     # Network Architecture is exactly same as in infoGAN (https://arxiv.org/abs/1606.03657)
#     # Architecture : (64)4c2s-(128)4c2s_BL-FC1024_BL-FC1_S
#     def __init__(self, input_dim=1, output_dim=1, image_size=32):
#         super(Discriminator, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.image_size = image_size
#
#         self.conv = nn.Sequential(
#             nn.Conv2d(self.input_dim, 64, 4, 2, 1), # 64, 16, 16
#             nn.LeakyReLU(0.2),
#             nn.Conv2d(64, 128, 4, 2, 1),  # 128, 8, 8
#             nn.BatchNorm2d(128),
#             nn.LeakyReLU(0.2),
#         )
#         self.fc = nn.Sequential(
#             nn.Linear(128 * (self.image_size // 4) * (self.image_size // 4), 1024),
#             nn.BatchNorm1d(1024),
#             nn.LeakyReLU(0.2),
#             nn.Linear(1024, self.output_dim),
#             nn.Sigmoid(),
#         )
#         initialize_weights(self)
#
#     def forward(self, input):
#         x = self.conv(input)
#         x = x.view(-1, 128 * (self.image_size // 4) * (self.image_size // 4))
#         x = self.fc(x)
#
#         return x


# Generator
Generator = nn.Sequential(
    nn.Linear(noise_dim, hidden_size),
    nn.ReLU(True),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(True),
    nn.Linear(hidden_size, image_size*image_size),
    nn.Tanh())


# Discriminator
Discriminator = nn.Sequential(
    nn.Linear(image_size*image_size, hidden_size),
    nn.LeakyReLU(0.2, True),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2, True),
    nn.Linear(hidden_size, 1),
    nn.Sigmoid())


# Device setting
# G = Generator(input_dim=noise_dim, output_dim=1, image_size=image_size).to(device)
# D = Discriminator(input_dim=1, output_dim=1, image_size=image_size).to(device)

G = Generator.to(device)
D = Discriminator.to(device)


# Binary cross entropy loss and optimizer
criterion = nn.BCELoss()
g_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate_g)
d_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate_d)

def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()


# Start training
total_step = len(data_loader)
for epoch in range(epochs):
    for i, (images, _) in enumerate(data_loader):
        _batch = images.size(0)
        images = images.reshape(_batch, -1).to(device)
        # images = images.to(device)

        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones(_batch, 1).to(device)
        fake_labels = torch.zeros(_batch, 1).to(device)

        # Labels shape is (batch_size, 1): [batch_size, 1]


        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        d_optimizer.zero_grad()

        outputs = D(images)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = torch.randn(_batch, noise_dim).to(device) # mean==0, std==1
        fake_images = G(z)

        outputs = D(fake_images.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        # Backprop and optimize
        d_loss = d_loss_real + d_loss_fake
        # reset_grad()
        d_loss.backward()
        d_optimizer.step()

        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #

        # Compute loss with fake images
        g_optimizer.zero_grad()

        z = torch.randn(_batch, noise_dim).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        gened_score = outputs

        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        g_loss = criterion(outputs, real_labels)

        # Backprop and optimize
        # reset_grad()
        g_loss.backward()
        g_optimizer.step()


        if (i + 1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                  .format(epoch+1, epochs, i + 1, total_step, d_loss.item(), g_loss.item(),
                          real_score.mean().item(), fake_score.mean().item()))


    result_images = denorm(G(fixed_noise)).detach().cpu()
    result_images = result_images.reshape(result_images.size(0), 1, 32, 32)
    result_images = make_grid(result_images).permute(1,2,0)
    print(result_images.size())
    plt.imshow(result_images.numpy())
    plt.show()


    # Save real images
    # if (epoch + 1) == 1:
    #     images = images.reshape(images.size(0), 1, 28, 28)
    #     save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))

    # tb.add_scalar(tag='d_loss', global_step=epoch+1, scalar_value=d_loss.item())
    # tb.add_scalar(tag='g_loss', global_step=epoch+1, scalar_value=g_loss.item())
    # tb.add_scalar(tag='real_score', global_step=epoch+1, scalar_value=real_score.mean().item())
    # tb.add_scalar(tag='fake_score', global_step=epoch+1, scalar_value=fake_score.mean().item())
    #
    # result_images = denorm(G(fixed_noise))
    # result_images = result_images.reshape(result_images.size(0), 1, 28, 28)
    # tb.add_images(tag='gened_images', global_step=epoch+1, img_tensor=result_images)

    # Save sampled images
    # fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    # save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch + 1)))

# Save the model checkpoints
# torch.save(G.state_dict(), 'G.ckpt')
# torch.save(D.state_dict(), 'D.ckpt')