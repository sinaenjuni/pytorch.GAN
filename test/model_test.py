import torch
import torch.nn as nn

from torch.nn.utils import spectral_norm
from models.resnet import resnet18
from torchvision.datasets import CIFAR10

from torchmetrics.functional import accuracy, confusion_matrix

dataset = CIFAR10("~/data")
print(dataset.data)


model = resnet18(sp=True)


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



inputs = torch.rand((1, 3, 64, 64))
outputs = model(inputs)
print(outputs[0].size())

for name, parma in model.named_parameters():
    print(name)

