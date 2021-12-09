import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def getInputModule(self, input, output):
        return nn.Linear(input, output)

    def getConvModule(self, input, output, kernel_size, stride, padding, bias):
        return nn.Sequential(
            nn.ConvTranspose2d(input, output, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(output),
            nn.ReLU(True)
        )
    def getOutputModule(self, input, output, kernel_size, stride, padding, bias):
        return nn.Sequential(
            nn.ConvTranspose2d(input, output, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.Tanh()
        )

    def __init__(self, nz, nc, nf = 48):
        super(Generator, self).__init__()

        self.nz = nz
        self.nc = nc
        self.nf = nf
        self.input =   self.getInputModule(nz,   nf*8)
        self.tconv1 =   self.getConvModule(nf*8, nf*4, 4, 1, 0, False)
        self.tconv2 =   self.getConvModule(nf*4, nf*2, 4, 2, 1, False)
        self.tconv3 =   self.getConvModule(nf*2, nf,   4, 2, 1, False)
        self.output = self.getOutputModule(nf,   nc,   4, 2, 1, False)

    def forward(self, x):
        input = self.input(x)
        input = input.view(-1, self.nf*8, 1, 1)
        tconv1 = self.tconv1(input)
        tconv2 = self.tconv2(tconv1)
        tconv3 = self.tconv3(tconv2)
        output = self.output(tconv3)
        return output


if __name__ == "__main__":

    G = Generator(100, 3, 48)
    input = torch.randn((64, 100))
    output = G(input)
    print(output.size())
    print("is main")


class Discriminator(nn.Module):
    def getInputModule(self, input, output, kernel_size, stride, padding, bias=False):
        return nn.Sequential(
            nn.Conv2d(input, output, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False)
        )

    def getConvModule(self, input, output, kernel_size, stride, padding, bias=False):
        return nn.Sequential(
            nn.Conv2d(input, output, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            nn.BatchNorm2d(output),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5, inplace=False)
        )

    def getOutputModule(self, input, ncls):
        adv = nn.Sequential(
            nn.Linear(input, 1),
            nn.Sigmoid()
        )
        cls = nn.Sequential(
            nn.Linear(input, ncls),
            nn.LogSoftmax(dim=1)
        )
        return adv, cls

    def __init__(self, nc, ncls, nf=16):
        super(Discriminator, self).__init__()
        self.nc = nc
        self.nf = nf
        self.ncls = ncls

        self.input =   self.getInputModule(nc,    nf,    3, 2, 1, False)
        self.conv1 =    self.getConvModule(nf,    nf*2,  3, 1, 1, False)
        self.conv2 =    self.getConvModule(nf*2,  nf*4,  3, 2, 1, False)
        self.conv3 =    self.getConvModule(nf*4,  nf*8,  3, 1, 1, False)
        self.conv4 =    self.getConvModule(nf*8,  nf*16, 3, 2, 1, False)
        self.conv5 =    self.getConvModule(nf*16, nf*32, 3, 1, 1, False)
        self.adv, self.cls = self.getOutputModule(nf*32*4*4, ncls)

    def forward(self, x):
        input = self.input(x)
        conv1 = self.conv1(input)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        conv5 = self.conv5(conv4)
        flat = conv5.view(conv5.size(0), -1)
        adv = self.adv(flat)
        cls = self.cls(flat)
        return adv, cls

if __name__ == "__main__":
    nc=3
    cls=10
    nf=16

    input = torch.randn((64, 3, 32, 32))
    labels = (torch.rand((64))*10).long()
    D=Discriminator(nc, cls, nf)
    adv, cls = D(input)
    print(adv.size(), cls.size())
    g_loss_cls = F.nll_loss(cls, labels)
    print(g_loss_cls)

    d_optimizer = torch.optim.Adam(D.parameters(), lr=0.002, betas=(0.5, 0.5))
    g_loss_cls.backward()
    # d_optimizer.step()


