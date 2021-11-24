import torchvision
from torchvision import transforms
from .data import getSubDataset
import numpy as np

class CIFAR10:
    def __init__(self):
        self.classes = {'plane': 0,
                        'car': 1,
                        'bird': 2,
                        'cat': 3,
                        'deer': 4,
                        'dog': 5,
                        'frog': 6,
                        'horse': 7,
                        'ship': 8,
                        'truck': 9}
        # Image processing
        transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize(mean=(0.5, 0.5, 0.5),   # 3 for RGB channels
                                             std=(0.5, 0.5, 0.5))])
        # Dataset define
        self.train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                           train=True,
                                           transform=transform,
                                           download=True)
        self.test_dataset = torchvision.datasets.CIFAR10(root='../../data/',
                                           train=False,
                                           transform=transform,
                                           download=True)
        labels = np.array(list(self.train_dataset.train_labels))
        ratio = [0.5 ** i for i in range(len(self.classes))]
        print(self.classes)
        print(labels)
        print(ratio)
        self.transformed_dataset, self.count = getSubDataset(dataset=self.train_dataset,
                                                   class_index=self.classes,
                                                   labels=labels,
                                                   lratio=ratio)
    def getTrainDataset(self):
        return self.train_dataset
    def getTransformedDataset(self):
        return self.transformed_dataset, self.count
    def getTestDataset(self):
        return self.test_dataset


class MNIST:
    def __init__(self, image_size=32):
        self.classes = {'0': 0,
                        '1': 1,
                        '2': 2,
                        '3': 3,
                        '4': 4,
                        '5': 5,
                        '6': 6,
                        '7': 7,
                        '8': 8,
                        '9': 9}

        self.transform = transforms.Compose([transforms.Resize(image_size),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean=[0.5],  # 1 for greyscale channels
                                                             std=[0.5])])
        # MNIST dataset
        self.train_dataset = torchvision.datasets.MNIST(root='../../data/',
                                                   train=True,
                                                   transform=self.transform,
                                                   download=True)
        # MNIST dataset
        self.test_dataset = torchvision.datasets.MNIST(root='../../data/',
                                                   train=False,
                                                   transform=self.transform,
                                                   download=True)

        labels = np.array(list(self.train_dataset.train_labels))
        ratio = [0.5 ** i for i in range(len(self.classes))]
        print(self.classes)
        print(labels)
        print(ratio)
        self.transformed_dataset, self.count = getSubDataset(dataset=self.train_dataset,
                                                   class_index=self.classes,
                                                   labels=labels,
                                                   lratio=ratio)
    def getTrainDataset(self):
        return self.train_dataset
    def getTransformedDataset(self):
        return self.transformed_dataset, self.count
    def getTestDataset(self):
        return self.test_dataset

if __name__ == '__main__':
    mnist = MNIST()
    train_dataset = mnist.getTrainDataset()
    train_dataset, count = mnist.getTransformedDataset()
    test_dataset = mnist.getTestDataset()

    print(count)

    cifar10 = CIFAR10()
    train_dataset = cifar10.getTrainDataset()
    transforms_dataset, count = cifar10.getTransformedDataset()
    test_dataset = cifar10.getTestDataset()

    print(count)