import torch
import torch.nn as nn

from torch.nn.utils import spectral_norm
from models.resnet import resnet18
from torchvision.datasets import CIFAR10

from torchmetrics.functional import accuracy, confusion_matrix

dataset = CIFAR10("~/data")
print(dataset.data)

inputs = torch.rand((128, 3, 32, 32))

x = inputs
model = resnet18()
for module in list(model.children())[:-2]:
    print("="*30)
    print(module)
    y = module(x)
    print(y.size())
    x = y



class FcNAdvModuel(nn.Module):
    def __init__(self, linear, feature, num_classes):
        super(FcNAdvModuel, self).__init__()
        self.flatten = nn.Flatten(1)
        self.cls = linear(in_features=feature, out_features=num_classes)
        self.adv = linear(in_features=feature, out_features=1)

    def forward(self, x):
        x = self.flatten(x)
        return self.adv(x), self.cls(x)
        # return self.adv(x)

model = nn.Sequential(*list(resnet18().children())[:-2])
model.add_module("last", FcNAdvModuel(nn.Linear, 512, 10))
print(model)
print(model(inputs)[0].size(), model(inputs)[1].size())
print(model(inputs).size())



model.add_module("flatten", nn.Flatten())
model.flatten.output
model.add_module("pool", nn.AdaptiveAvgPool2d((1,1)))


model.add_module('fc', FcNAdvModuel(nn.Linear, 10))

print(model)

model.fc = nn.Linear(in_features=512, out_features=10)
print(model)

F.softmax(label_logit, dim=-1)


for name, parma in model.named_parameters():
    print(name)




class FcNAdvModuel(nn.Module):
    def __init__(self):
        super(FcNAdvModuel, self).__init__()
        self.fc = spectral_norm(nn.Linear(in_features=512, out_features=10))
        self.adv = nn.Linear(in_features=512, out_features=1)

    def forward(self, x):
        return self.fc(x), self.adv(x)

model.fc = FcNAdvModuel()



outputs = model(inputs)
# print(outputs[0].size())

# for name, parma in model.named_parameters():
#     print(name)

