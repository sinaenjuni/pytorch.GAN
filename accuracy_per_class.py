from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, Normalize, ToTensor

import numpy as np

transforms = Compose([Resize(32),
                      ToTensor(),
                      Normalize(0.5, 0.5),
                      ])

dataset = MNIST("./data", train=True, transform=transforms, download=False)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

print(dataset)
print(loader)

target_np = np.array([])
predict_np = np.array([])
# for i, (image, target) in enumerate(loader):
#     target = target.cuda()
#
#     # target_np = np.append(target_np, target.item().)
#     # print(target.cpu().numpy())
#
#     target_np = np.append(target_np, target.cpu().numpy())
#
# print(target_np)


from sklearn.metrics import confusion_matrix
t = np.array([0,0,1,0])
t = (np.random.rand(10) * 3).astype(np.long)
p = np.array([1,0,1,0])
p = (np.random.rand(10) * 3).astype(np.long)
conf = confusion_matrix(t, p)
print(conf)

for i, _conf in enumerate(conf):
    _acc = _conf[i] / _conf.sum()
    print(_acc)
print(conf.sum())
acc = np.trace(conf) / conf.sum()
print(acc)
# pos = t == p
# unique, count = np.unique(t, return_counts=True)
# for _unique, _count in zip(unique, count):
#     ind = np.where( t==_unique )[0]
#     _pos = pos[ind].sum()
#     acc = _pos/_count
#     # print(_unique, _count)
#     print(acc)
# print(t[ind])
# print(pos[ind].sum())