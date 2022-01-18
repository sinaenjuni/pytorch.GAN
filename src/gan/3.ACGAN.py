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
num_class = 10
hidden_size = 256
batch_size = 64

epochs = 200
learning_rate_g = 0.0002
learning_rate_d = 0.0002
beta1 = 0.5
beta2 = 0.999
sample_dir = '../samples'

fixed_noise = torch.randn(10, noise_dim).to(device).repeat(10, 1)

index = torch.tensor([[i//10] for i in range(100)])
fixed_onehot = torch.zeros(100, 10).scatter_(1, index, 1).to(device)

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
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear(noise_dim + num_class, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, image_size*image_size),
            nn.Tanh())

    def forward(self, x):
        return self.layer(x)


# Discriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.layer = nn.Sequential(
            nn.Linear((image_size * image_size), hidden_size),
            nn.LeakyReLU(0.2, True),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(0.2, True))

        self.adv = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Sigmoid())

        self.cls = nn.Sequential(nn.Linear(hidden_size, 10))

    def forward(self, x):
        out = self.layer(x)
        adv = self.adv(out)
        cls = self.cls(out)
        return adv, cls



# Device setting
# G = Generator(input_dim=noise_dim, output_dim=1, image_size=image_size).to(device)
# D = Discriminator(input_dim=1, output_dim=1, image_size=image_size).to(device)

G = Generator().to(device)
D = Discriminator().to(device)


# Binary cross entropy loss and optimizer
BCE_loss = nn.BCELoss()
CE_loss = nn.CrossEntropyLoss()

g_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate_g)
d_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate_d)


def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()

onehot = torch.eye(10).to(device)


# Start training
total_step = len(data_loader)
for epoch in range(epochs):
    for i, (images, target) in enumerate(data_loader):
        _batch = images.size(0)
        images = images.reshape(_batch, -1).to(device)
        target = target.to(device)
        # images = images.to(device)
        onehot_target = onehot[target]


        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones(_batch, 1).to(device)
        fake_labels = torch.zeros(_batch, 1).to(device)

        # Labels shape is (batch_size, 1): [batch_size, 1]


        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        # images = torch.cat([images, target], 1)
        d_real_avd, d_real_cls = D(images)
        d_real_avd_loss = BCE_loss(d_real_avd, real_labels)
        d_real_cls_loss = CE_loss(d_real_cls, target)

        real_score = d_real_avd

        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = torch.randn(_batch, noise_dim).to(device) # mean==0, std==1
        z = torch.cat([z, onehot_target], 1)
        fake_images = G(z)

        # outputs = D(fake_images.detach())
        # outputs = D(torch.cat([fake_images.detach(), target], 1))
        d_fake_avd, d_fake_cls = D(fake_images.detach())
        d_fake_avd_loss = BCE_loss(d_fake_avd, fake_labels)
        d_fake_cls_loss = CE_loss(d_fake_cls, target)
        fake_score = d_fake_avd

        # Backprop and optimize
        d_loss = d_real_avd_loss + d_fake_avd_loss + d_real_cls_loss + d_fake_cls_loss

        reset_grad()
        d_loss.backward()
        d_optimizer.step()

        # ================================================================== #
        #                        Train the generator                         #
        # ================================================================== #

        # Compute loss with fake images
        z = torch.randn(_batch, noise_dim).to(device)
        z = torch.cat([z, onehot_target], 1)
        fake_images = G(z)
        # fake_images = torch.cat([fake_images, target], 1)
        # outputs = D(fake_images)
        g_fake_avd, g_fake_cls = D(fake_images)
        g_fake_avd_loss = BCE_loss(g_fake_avd, real_labels)
        g_fake_cls_loss = CE_loss(g_fake_cls, target)
        gened_score = g_fake_avd

        g_loss = g_fake_avd_loss + g_fake_cls_loss
        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf

        # Backprop and optimize
        reset_grad()
        g_loss.backward()
        g_optimizer.step()


        if (i + 1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                  .format(epoch+1, epochs, i + 1, total_step, d_loss.item(), g_loss.item(),
                          real_score.mean().item(), fake_score.mean().item()))


    result_images = denorm(G(torch.cat([fixed_noise, fixed_onehot], 1))).detach().cpu()
    # result_images = denorm(G(fixed_noise)).detach().cpu()
    result_images = result_images.reshape(result_images.size(0), 1, 32, 32)
    result_images = make_grid(result_images, nrow=10).permute(1,2,0)
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