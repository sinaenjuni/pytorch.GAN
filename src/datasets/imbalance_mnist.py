# From: https://github.com/kaidic/LDAM-DRW
import torch
from torchvision.datasets import MNIST
from torch.utils.data import Sampler, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize
import numpy as np
import random
from datasets.sampler import BalancedSampler


class Imbalanced_MNIST(MNIST):
    cls_num = 10

    def __init__(self, root, imb_type='exp', train=True,imb_factor=0.01, rand_number=0,
                 transform=None, target_transform=None,
                 download=False, reverse=False):
        super(Imbalanced_MNIST, self).__init__(root=root, transform=transform, target_transform=target_transform, download=download, train=train)
        np.random.seed(rand_number)
        img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor, reverse)
        self.gen_imbalanced_data(img_num_list)
        self.reverse = reverse

    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor, reverse):
        img_max = len(self.data) / cls_num
        img_num_per_cls = []

        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                if reverse:
                    num = img_max * (imb_factor ** (((cls_num - 1) - cls_idx) / (cls_num - 1.0)))
                    img_num_per_cls.append(int(num))
                else:
                    num = img_max * (imb_factor ** (cls_idx / (cls_num - 1.0)))
                    img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)
        return img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls):
        new_data = []
        new_targets = []

        targets_np = np.array(self.targets, dtype=np.int64)

        classes = np.unique(targets_np)
        # np.random.shuffle(classes)
        self.num_per_cls_dict = dict()
        for the_class, the_img_num in zip(classes, img_num_per_cls):
            self.num_per_cls_dict[the_class] = the_img_num
            idx = np.where(targets_np == the_class)[0]
            # np.random.shuffle(idx)
            selec_idx = idx[:the_img_num]
            new_data.append(self.data[selec_idx, ...])
            # new_targets.extend([the_class, ] * the_img_num)
            new_targets.extend(targets_np[selec_idx])

        new_data = np.vstack(new_data)
        self.data = torch.tensor(new_data)
        self.targets = torch.tensor(new_targets)

    def get_cls_num_list(self):
        cls_num_list = []
        for i in range(self.cls_num):
            cls_num_list.append(self.num_per_cls_dict[i])
        return cls_num_list



if __name__ == '__main__':
    from PIL import Image

    transform = Compose([ToTensor(),
                         Normalize([0.5], [0.5])
                         ])

    dataset = Imbalanced_MNIST(root='~/data/', imb_factor=0.01,
                               download=True, transform=transform)


    balancedSampler = BalancedSampler(dataset, retain_epoch_size=False)
    dataloader = DataLoader(dataset, batch_size=10, sampler=balancedSampler)

    count = [0 for _ in range(10)]
    for data in dataloader:
        # print(data_module[1])
        for i in data[1]:
            count[i] += 1
    print(count)