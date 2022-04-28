import torch
import torch.nn as nn


class Decoder_module(nn.Module):
    def getLayer(self, num_input, num_outout, kernel_size, stride, padding, norm):
        tconv = nn.ConvTranspose2d(in_channels=num_input,
                                       out_channels=num_outout,
                                       kernel_size=kernel_size,
                                       stride=stride,
                                       padding=padding)
        if norm == 'bn':
            layer = nn.Sequential(tconv,
                          nn.BatchNorm2d(num_outout),
                          nn.LeakyReLU(negative_slope=0.2, inplace=True))
        elif norm == 'sn':
            layer = nn.Sequential(tconv,
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True))
        else:
            layer = nn.Sequential(tconv,
                                  nn.LeakyReLU(negative_slope=0.2, inplace=True))
        return layer

    def __init__(self, image_size, image_channel, std_channel, latent_dim, norm):
        super(Decoder_module, self).__init__()

        self.image_size = image_size // 2 ** 4
        self.std_channel = std_channel

        self.layer1 = nn.Sequential(nn.Linear(in_features=latent_dim,
                                              out_features = self.image_size * self.image_size * std_channel * 4),
                                    nn.LeakyReLU(negative_slope=0.2, inplace=True))  # 2*2*256

        self.layer2 = self.getLayer(std_channel*4, std_channel*2, kernel_size=4, stride=2, padding=1, norm=norm)   # 4*4*128
        self.layer3 = self.getLayer(std_channel*2, std_channel*2, kernel_size=4, stride=2, padding=1, norm=norm)   # 8*8*128
        self.layer4 = self.getLayer(std_channel*2, std_channel*1, kernel_size=4, stride=2, padding=1, norm=norm)   # 16*16*64
        self.layer5 = nn.Sequential(nn.ConvTranspose2d(std_channel*1, image_channel, kernel_size=4, stride=2, padding=1))   # 32*32*3

    def forward(self, x):
        x = self.layer1(x)
        x = x.view(-1, self.std_channel * 4, self.image_size, self.image_size)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = torch.tanh_(x)
        # x = torch.sigmoid_(x)
        return x


if __name__ == "__main__":
    def initialize_weights(m):
        if isinstance(m, nn.Conv2d):
            nn.init.normal_(m.weight.data, std=0.02)
        if isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight.data, std=0.02)

    D = Decoder_module(image_size=64, image_channel=3, std_channel=64, latent_dim=128, norm='bn')
    D.apply(initialize_weights)

    inputs = torch.randn((1, 128))
    outputs = D(inputs)
    print(outputs.size())

    for i in D.named_parameters():
        print(i[0])