import torch.nn as nn


# Discriminator
class Discriminator(nn.Module):
    def __init__(self, nc, ndf, ngpu):
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
            # nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False)
            # nn.Sigmoid()
        )
        self.adv = nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False)
        self.cls = nn.Conv2d(ndf * 4, 10, 4, 1, 0, bias=False)

    def forward(self, input):
        y = self.main(input)
        adv = self.adv(y)
        cls = self.cls(y)
        return adv, cls
        # return self.main(input)

# Generator
class Generator(nn.Module):
    def __init__(self, nz, nc, ngf, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            # nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            # nn.BatchNorm2d(ngf * 8),
            # nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(nz, ngf * 4, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


if __name__ == '__main__':
    import torch
    import torch.nn.functional as F

    nc = 3
    ndf = 32

    nz = 100
    ngf = 32

    D = Discriminator(nc, ndf, ngpu=1)
    G = Generator(nz, ngf, ngpu=1)

    batch_size = 64
    img_size = 32

    img = torch.randn((batch_size, nc, img_size, img_size))
    labels = torch.randint(10, (batch_size,))
    z = torch.randn((batch_size, nz, 1, 1))

    print(img.size())
    print(z.size())
    d_out_adv, d_out_cls = D(img)
    g_out = G(z)

    print(d_out_adv.size(), d_out_cls.size())
    print(g_out.size())

    d_out_adv = d_out_adv.view(-1)
    d_out_cls = d_out_cls.view(-1, 10)
    print(d_out_adv.size(), d_out_cls.size())

    cls_loss = F.cross_entropy(d_out_cls, labels)
    print(cls_loss)
