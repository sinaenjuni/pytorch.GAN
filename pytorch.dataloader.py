import torch
import numpy as np
from utiles.imbalance_mnist import IMBALANCEMNIST
from torchvision import transforms
from torch.utils.data import Dataset, ConcatDataset, Sampler, RandomSampler, BatchSampler, DataLoader, SequentialSampler

transform = transforms.Compose([
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5],  # 1 for greyscale channels
                         std=[0.5])])


class SelectSampler(Sampler):
    def __init__(self, data_source, target_label):
        self.data_source = data_source
        print("adawD", len(data_source.targets))
        self.target_idx = np.where(data_source.targets == target_label)[0]
        print(self.target_idx)
    def __iter__(self):
        return iter(self.target_idx)
    def __len__(self):
        return len(self.target_idx)

mnist = IMBALANCEMNIST(root='./data/',
                           train=True,
                           transform=transform,
                           download=False,
                           imb_factor=0.01)

print("data", len(mnist))
print("img", len(mnist.data))
print("label", len(mnist.train_labels))
select_sampler = SelectSampler(mnist, 9)
loader = DataLoader(mnist, batch_size=32, sampler=select_sampler)

for i in loader:
    print(i)
# print(len(loader))

# class MyDataset(Dataset):
#     def __init__(self):
#         self.data = np.empty((0, 3))
#         self.label = np.empty((0, 1))
#
#         for idx in range(10):
#             self.data = np.vstack(self.data, np.array([idx, 2 * idx, 3 * idx]))
#             self.label = np.append(self.label, np.array([[idx]]), axis=0)
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         return {'input': self.data,
#                 'label': self.label}
#
#
# dataset = MyDataset()

# print("기본 Data Loader")
# loader = DataLoader(dataset)
# for data in loader:
#     print(data['input'])
#     print(data['label'])
#
# print("Data Loader with Batch size")
# loader = DataLoader(dataset, batch_size=4)
# for data in loader:
#     print(data['input'])
#     print(data['label'])
#
# print("Data Loader with Random Sampler")
# point_sampler = RandomSampler(dataset)
# loader = DataLoader(dataset, batch_size=4, sampler=point_sampler)
# for data in loader:
#     print(data['input'])
#     print(data['label'])

# t = torch.Tensor([1, 2, 3])
# print((t == 2).nonzero(as_tuple=False)[0])
#
# class SelectSampler(Sampler):
#     def __init__(self, data_source, target_label):
#         self.data_source = data_source
#         self.target_idx = (data_source == target_label).nonzero(as_tuple=False)
#
#     def __iter__(self):
#         return iter(self.target_idx)
#     def __len__(self):
#         return len(self.data_source)
#
#
# # select_sampler = SelectSampler(dataset, 1)
# loader = DataLoader(dataset, batch_size=1)
# for data in loader:
#     print(data['input'])
#     print(data['label'])

# print("Data Loader with Batch Sampler")
# point_sampler = RandomSampler(dataset)
# batch_sampler = BatchSampler(point_sampler, 1, False)
# loader = DataLoader(dataset, sampler=batch_sampler)
# for data in loader:
#     print(data['input'])
#     print(data['label'])


# class VarMapDataset(Dataset):
#     def __len__(self):
#         return 10
#     def __getitem__(self, idx):
#         return {"input": torch.tensor([idx] * (idx + 1),
#                                       dtype=torch.float32),
#                 "label": torch.tensor(idx,
#                                       dtype=torch.float32)}
#
# var_map_dataset = VarMapDataset()
#
#
# def make_batch(samples):
#     inputs = [sample['input'] for sample in samples]
#     labels = [sample['label'] for sample in samples]
#     padded_inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first=True)
#     return {'input': padded_inputs.contiguous(),
#             'label': torch.stack(labels).contiguous()}
#
# loader = DataLoader(var_map_dataset, batch_size=4, collate_fn=make_batch)
# for data in loader:
#     print(data['input'])
#     print(data['label'])



