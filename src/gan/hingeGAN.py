import os
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.tensorboard import SummaryWriter

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:', device)

# TensorBoard define
log_dir = '../tb_logs/hingeGAN/test1'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
tb = SummaryWriter(log_dir=log_dir)

# Hyper-parameters
latent_size = 64
hidden_size = 256
image_size = 28*28
num_epochs = 200
batch_size = 100
learning_rate = 0.0002

fixed_noise = torch.randn(batch_size, latent_size).to(device)

# Create a directory if not exists
# if not os.path.exists(sample_dir):
#     os.makedirs(sample_dir)

def denorm(x):
    out = (x + 1) / 2
    return out.clamp(0, 1)

# Image processing
# transform = transforms.Compose([
#                 transforms.ToTensor(),
#                 transforms.Normalize(mean=(0.5, 0.5, 0.5),   # 3 for RGB channels
#                                      std=(0.5, 0.5, 0.5))])

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],  # 1 for greyscale channels
                         std=[0.5])])

# MNIST dataset
mnist = torchvision.datasets.MNIST(root='../data/',
                                   train=True,
                                   transform=transform,
                                   download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=mnist,
                                          batch_size=batch_size,
                                          shuffle=True)


# Discriminator
D = nn.Sequential(
    nn.Linear(image_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, hidden_size),
    nn.LeakyReLU(0.2),
    nn.Linear(hidden_size, 1)
    # nn.Sigmoid()
    )

# Generator
G = nn.Sequential(
    nn.Linear(latent_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, image_size),
    nn.Tanh())

# Device setting
D = D.to(device)
G = G.to(device)

# Binary cross entropy loss and optimizer
criterion = nn.BCELoss()
d_optimizer = torch.optim.Adam(D.parameters(), lr=learning_rate)
g_optimizer = torch.optim.Adam(G.parameters(), lr=learning_rate)

def reset_grad():
    d_optimizer.zero_grad()
    g_optimizer.zero_grad()


# Start training
total_step = len(data_loader)
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(data_loader):
        images = images.reshape(batch_size, -1).to(device)

        # Create the labels which are later used as input for the BCE loss
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        # Labels shape is (batch_size, 1): [batch_size, 1]


        # ================================================================== #
        #                      Train the discriminator                       #
        # ================================================================== #

        # Compute BCE_Loss using real images where BCE_Loss(x, y): - y * log(D(x)) - (1-y) * log(1 - D(x))
        # Second term of the loss is always zero since real_labels == 1
        outputs = D(images)
        # d_loss_real = criterion(outputs, real_labels)
        d_loss_real = F.relu(1. - outputs).mean()
        real_score = outputs

        # Compute BCELoss using fake images
        # First term of the loss is always zero since fake_labels == 0
        z = torch.randn(batch_size, latent_size).to(device) # mean==0, std==1
        fake_images = G(z)

        outputs = D(fake_images.detach())
        # d_loss_fake = criterion(outputs, fake_labels)
        d_loss_fake = F.relu(1. + outputs).mean()
        fake_score = outputs

        # Backprop and optimize
        d_loss = d_loss_real + d_loss_fake
        reset_grad()
        d_loss.backward()
        d_optimizer.step()

        # ================================================================== #
        #                        Train the generator                         #
        # https://arxiv.org/pdf/1802.05957.pdf                               #
        # ================================================================== #


        # Compute loss with fake images
        z = torch.randn(batch_size, latent_size).to(device)
        fake_images = G(z)
        outputs = D(fake_images)
        gened_score = outputs

        # We train G to maximize log(D(G(z)) instead of minimizing log(1-D(G(z)))
        # For the reason, see the last paragraph of section 3. https://arxiv.org/pdf/1406.2661.pdf
        # g_loss = criterion(outputs, real_labels)
        g_loss = -outputs.mean()

        # Backprop and optimize
        reset_grad()
        g_loss.backward()
        g_optimizer.step()


        if (i + 1) % 200 == 0:
            print('Epoch [{}/{}], Step [{}/{}], d_loss: {:.4f}, g_loss: {:.4f}, D(x): {:.2f}, D(G(z)): {:.2f}'
                  .format(epoch+1, num_epochs, i + 1, total_step, d_loss.item(), g_loss.item(),
                          real_score.mean().item(), fake_score.mean().item()))

    # Save real images
    # if (epoch + 1) == 1:
    #     images = images.reshape(images.size(0), 1, 28, 28)
    #     save_image(denorm(images), os.path.join(sample_dir, 'real_images.png'))

    tb.add_scalar(tag='d_loss', global_step=epoch+1, scalar_value=d_loss.item())
    tb.add_scalar(tag='g_loss', global_step=epoch+1, scalar_value=g_loss.item())
    tb.add_scalar(tag='real_score', global_step=epoch+1, scalar_value=real_score.mean().item())
    tb.add_scalar(tag='fake_score', global_step=epoch+1, scalar_value=fake_score.mean().item())

    result_images = denorm(G(fixed_noise))
    result_images = result_images.reshape(result_images.size(0), 1, 28, 28)
    tb.add_images(tag='gened_images', global_step=epoch+1, img_tensor=result_images)

    # Save sampled images
    # fake_images = fake_images.reshape(fake_images.size(0), 1, 28, 28)
    # save_image(denorm(fake_images), os.path.join(sample_dir, 'fake_images-{}.png'.format(epoch + 1)))

# Save the model checkpoints
# torch.save(G.state_dict(), 'G.ckpt')
# torch.save(D.state_dict(), 'D.ckpt')