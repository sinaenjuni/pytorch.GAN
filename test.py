import torch
import torch.nn.functional as F

#
# x = [[9, 5, 1],
#      [7, 9, 1],
#      [5, 4, 9]]
#
# y = [2,
#      1,
#      2]
#
# x = torch.FloatTensor(x)
# y = torch.LongTensor(y)
#
# y_ = torch.zeros(len(y),len(y))
# y_.scatter_(1, y.unsqueeze(1), 1)
# print(y_)
# print((y_*-torch.log(F.softmax(x, dim=1))).sum(1).mean())
# # F.softmax() + torch.log() = F.log_softmax()
# print((y_*-F.log_softmax(x, dim=1)).sum(1).mean())
# # F.log_softmax() + F.nll_loss() = F.cross_entropy()
# print(F.nll_loss(F.log_softmax(x, dim=1), y))
# print(F.cross_entropy(x, y))
#
#
#
#
# weight = torch.tensor([0.,
#                        100.,
#                        100.])
#
# weighted_ce = F.cross_entropy(x, y, weight=weight)
# print("weighted_ce", weighted_ce)
#
#
# img_size = 32
# fill = torch.zeros([10, 10, img_size, img_size])
# for i in range(10):
#     fill[i, i, :, :] = 1
#
# print(fill[1].mean())



# # fixed noise & label
# temp_z_ = torch.randn(10, 100)
# fixed_z_ = temp_z_
# fixed_y_ = torch.zeros(10, 1)
# for i in range(9):
#     fixed_z_ = torch.cat([fixed_z_, temp_z_], 0)
#     temp = torch.ones(10, 1) + i
#     fixed_y_ = torch.cat([fixed_y_, temp], 0)
#
# print(fixed_z_.size())
# print(fixed_y_.size())
# print(fixed_y_[:11])
#
# fixed_z_ = fixed_z_.view(-1, 100, 1, 1)
# fixed_y_label_ = torch.zeros(100, 10)
# fixed_y_label_.scatter_(1, fixed_y_.type(torch.LongTensor), 1)
# fixed_y_label_ = fixed_y_label_.view(-1, 10, 1, 1)
#
#
# img_size = 32
# # label preprocess
# onehot = torch.zeros(10, 10)
# onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(10,1), 1).view(10, 10, 1, 1)
# print(onehot[[1,2]].size())
# fill = torch.zeros([10, 10, img_size, img_size])
# for i in range(10):
#     fill[i, i, :, :] = 1
#
# print(fill[1].mean((1,2)))
#
#
# mini_batch = 32
# y_ = (torch.rand(mini_batch, 1) * 10).type(torch.LongTensor).squeeze()
# print(y_)




# import numpy as np
#
# no_data = 10
# no_class = 5
# no_per_class = np.array([1,2,3,4,5])
#
# e = (1.0-no_data)/(1.0-no_data ** no_per_class)
# print(e)


# import torch
#
# r = (torch.rand((10,))*10).long()
# print(r)
#
# print(torch.nn.ConvTranspose2d(10,4,4,2,1).__class__.__name__)
#
# label_dim=10
# # label preprocess
# onehot = torch.zeros(label_dim, label_dim)
# onehot = onehot.scatter_(1, torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]).view(label_dim, 1), 1).view(label_dim, label_dim, 1, 1)
#
# print(onehot[0])
#
# print(torch.exp(torch.tensor([1.])))


#
# import torch
# label_dim = 10
# G_input_dim = 100
#
# temp_noise = torch.randn(label_dim, G_input_dim)
# fixed_noise = temp_noise
# fixed_c = torch.zeros(label_dim, 1)
# for i in range(9):
#     fixed_noise = torch.cat([fixed_noise, temp_noise], 0)
#     temp = torch.ones(label_dim, 1) + i
#     fixed_c = torch.cat([fixed_c, temp], 0)
#
# fixed_noise = fixed_noise.view(-1, G_input_dim, 1, 1)
# # for i in range(0,100,10):
# #     print(fixed_noise[i:i+10,:5].squeeze())
# print(fixed_c.size())
#
# fixed_label = torch.zeros(G_input_dim, label_dim)
# fixed_label.scatter_(1, fixed_c.type(torch.LongTensor), 1)
# fixed_label = fixed_label.view(-1, label_dim, 1, 1)
#
# print(fixed_noise.size())
# print(fixed_label.size())


# import torch
# import torch.nn.functional as F
#
#
# tensor = torch.tensor([[2, 2],
#                       [2, 2]]).float()
# print(tensor)
# renormed = torch.renorm(tensor, 1, 0, 2)
# print(renormed)
#
# print(tensor[1, :].sum())
# print(renormed[1, :].sum())
# print(tensor[:, 1].sum())
# print(renormed[:, 1].sum())
#
# factor = 2/6
# print([i * factor**2 for i in tensor[1,:]])

# import torch
# import torch.nn.functional as F
#
# x = torch.tensor([.8])
# h = torch.tensor([.2])
# w_xh = torch.tensor([.2], requires_grad=True)
# w_hh = torch.tensor([.3], requires_grad=True)
# b_h = torch.tensor([.9], requires_grad=True)
# h_t = torch.tanh(w_hh*h + w_xh*x + b_h)
# print(h_t)
# h_t.backward()
# print(w_xh.grad)

# out = torch.tanh(input)
# print(out)


# import torchvision
# class IMBALANCECIFAR10(torchvision.datasets.CIFAR10):
#     def __init__(self):
#         root="data"
#         train=True
#         transform = None
#         target_transform = None
#         download = True
#         super(IMBALANCECIFAR10, self).__init__(root, train, transform, target_transform, download)
#         print(len(self.data))
#
# cifar10 = IMBALANCECIFAR10()

# img_max=50000
# cls_num=10
# imb_factor=0.1
# img_num_per_cls=[]
# reverse = False
# for cls_idx in range(cls_num):
#     if reverse:
#         num = img_max * (imb_factor ** ((cls_num - 1 - cls_idx) / (cls_num - 1.0)))
#         img_num_per_cls.append(int(num))
#     else:
#         num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
#         img_num_per_cls.append(int(num))
# print(img_num_per_cls)
#

# import numpy as np
# print(np.power(2, 6))
#
# for i in range(1, 5):
#     print(i)



import numpy as np

classes = 10
counts = [ 5000 * (0.5 ** i) for i in range(classes)]
print(counts)

# majority = 5000+2500+1250+625+312
majority = 2500
# minority = 156+78+39+19+9
minority = 2500

non_weighted = [912, 873, 559, 411, 229, 126, 89, 52, 2, 0]
weighted =     [914, 890, 596, 475, 338, 183, 245, 103, 9, 3]
non_weighted = [926, 967, 691, 630, 660, 535, 550, 454, 288, 124]
weighted =     [918, 949, 739, 644, 554, 449, 588, 478, 324, 142]

non_weighted = np.array(non_weighted)
weighted = np.array(weighted)

print(sum(non_weighted))
print(sum(weighted))

# non_weighted_acc=0
# weighted_acc=0
# for i in range(0, 10, 1):
#     print(i)
#     non_weighted_acc += non_weighted[i]
#     weighted_acc += weighted[i]
#
# print(non_weighted_acc)
# print(weighted_acc)
# print((weighted_acc/5000)*100)
# print((non_weighted_acc/5000)*100)


# 데이터셋 개수 확인
# import numpy as np
# from utiles.imbalance_cifar import IMBALANCECIFAR10, IMBALANCECIFAR100
# from torch.utils.data import DataLoader
#
# cifar10_imbalance_dataset = IMBALANCECIFAR100("./data", train=True, download=False, transform=None, imb_factor=0.01)
# print(len(cifar10_imbalance_dataset.targets))
# print(cifar10_imbalance_dataset.classes)
# print(cifar10_imbalance_dataset.class_to_idx)
# print()
# targets = np.array(cifar10_imbalance_dataset.targets)
# unique, count = np.unique(targets, return_counts=True)
# print(dict(zip(unique, count)))

# cifar10_imbalance_loader = DataLoader(cifar10_imbalance_dataset, )

# print(cifar10_imbalance_dataset)
# for data in cifar10_imbalance_dataset:
#     img, label = data
#     print(label)





