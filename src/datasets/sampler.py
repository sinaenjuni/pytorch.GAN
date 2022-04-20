import torch
import random
import numpy as np
from torch.utils.data import Sampler, Dataset

class SelectSampler(Sampler):
    def __init__(self, dataset, target_label, shuffle):
        targets = torch.tensor(dataset.train_labels)
        self.target_idx = np.where(targets == target_label)[0]
        self.shuffle = shuffle
        self.n = len(self.target_idx)

    def __iter__(self):
        if self.shuffle:
            generator = torch.Generator()
            generator.manual_seed(int(torch.empty((), dtype=torch.int64).random_().item()))
            yield from self.target_idx[torch.randperm(self.n, generator=generator).tolist()]
        else:
            yield from self.target_idx

    def __len__(self):
        return self.n

class BalancedSampler(Sampler):
    def __init__(self, dataset, retain_epoch_size=False):

        targets = dataset.targets

        num_classes = len(np.unique(targets))
        cls_num_list = [0] * num_classes
        for label in targets:
            cls_num_list[label] += 1

        buckets = [[] for _ in range(num_classes)]
        for idx, label in enumerate(targets):
            buckets[label].append(idx)

        for bucket in buckets:
            random.shuffle(bucket)

        self.bucket_num = len(buckets)
        self.buckets = buckets
        self.bucket_pointers = [0 for _ in range(self.bucket_num)]
        self.retain_epoch_size = retain_epoch_size

    def __iter__(self):
        count = self.__len__()
        # print('sampler', count)
        while count > 0:
            yield self._next_item()
            count -= 1

    def _next_item(self):
        bucket_idx = random.randint(0, self.bucket_num - 1)
        bucket = self.buckets[bucket_idx]
        item = bucket[self.bucket_pointers[bucket_idx]]
        self.bucket_pointers[bucket_idx] += 1
        if self.bucket_pointers[bucket_idx] == len(bucket):
            self.bucket_pointers[bucket_idx] = 0
            random.shuffle(bucket)
        return item

    def __len__(self):
        if self.retain_epoch_size:
            return sum([len(bucket) for bucket in self.buckets])  # AcruQRally we need to upscale to next full batch
        else:
            return max([len(bucket) for bucket in
                        self.buckets]) * self.bucket_num  # Ensures every instance has the chance to be visited in an epoch



if __name__ == "__main__":
    from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
    from datasets.imbalance_fashion_mnist import Imbalanced_FashionMNIST
    from torch.utils.data import DataLoader
    from torchvision.transforms import *

    # dataset = MNIST(root="~/datasets/mnist", train=True, transform=None, target_transform=None, download=False)
    # select_sampler = SelectSampler(dataset, 1, False)
    #
    # data_loader = DataLoader(dataset, batch_size=1, shuffle=False, sampler=select_sampler)
    #
    # for i in data_loader:
    #     print(i)

    train_trsfm = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        # transforms.RandomRotation(15),
        Resize(64),
        ToTensor(),
        Normalize(mean=[0.5], std=[0.5])
    ])

    dataset = Imbalanced_FashionMNIST(root='~/data/',
                                      train=True,
                                      imb_factor=0.01,
                                      download=True,
                                      transform=train_trsfm)

    sampler = BalancedSampler(dataset, retain_epoch_size=False)
    loader = DataLoader(dataset=dataset, sampler=sampler, batch_size=10, shuffle=False)

    # print(len(loader))
    count = {i:0 for i in range(10)}
    for image, label in loader:
        for l in label.tolist():
            # print(l)
            count[l] += 1

    print(count)





