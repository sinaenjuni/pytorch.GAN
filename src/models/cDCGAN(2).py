import torch
import torch.nn as nn

def getCBR(in_channels, out_channels, kernel_size, stride, padding):
    return nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                            kernel_size=kernel_size, stride=stride, padding=padding),
                            nn.BatchNorm2d(out_channels),
                            nn.ReLU(True))

nf = 16

input_tensor = torch.rand((64, 100))
input_condition = torch.rand((64, 10))

input_layer = getCBR(100, nf*4, kernel_size=4, stride=1, padding=0)
condition_layer = getCBR(10, nf*4, kernel_size=4, stride=1, padding=0)

layer1 = getCBR(nf*4*2, 128, kernel_size=4, stride=2, padding=1)

x = input_layer(input_tensor.view(-1, 100, 1, 1))
print(x.size())
c = condition_layer(input_condition.view(-1, 10, 1, 1))
print(c.size())
cat = torch.cat([x,c],1)
print(cat.size())
out = layer1(cat)
print(out.size())


# out = layer1(torch.cat([x,c]))

#
# layer1 = nn.Sequential(
#     nn.ConvTranspose2d(in_channels=nf * 8, out_channels=nf * 4, kernel_size=4, stride=2, padding=0, bias=False),
#     nn.BatchNorm2d(nf * 4),
#     nn.ReLU(True))
# layer2 = nn.Sequential(
#     nn.ConvTranspose2d(in_channels=nf * 4, out_channels=nf * 2, kernel_size=4, stride=2, padding=1, bias=False),
#     nn.BatchNorm2d(nf * 2),
#     nn.ReLU(True))
# layer3 = nn.Sequential(
#     nn.ConvTranspose2d(in_channels=nf * 2, out_channels=nf * 1, kernel_size=4, stride=2, padding=1, bias=False),
#     nn.BatchNorm2d(nf * 1),
#     nn.ReLU(True))
# output_layer = nn.Sequential(
#     nn.ConvTranspose2d(in_channels=nf * 1, out_channels=nc, kernel_size=4, stride=2, padding=1, bias=False),
#     nn.Tanh())