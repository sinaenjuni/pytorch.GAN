from torchvision.transforms import Compose, ToTensor, Normalize, Resize
from torchvision.datasets import CIFAR10, MNIST, FashionMNIST
from torch.utils.data import DataLoader

def cifar(image_size=32, train=True, batch_size=64) -> DataLoader:
        transform = Compose([Resize(image_size),
                             ToTensor(),
                             # Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                             ])
        dataset = CIFAR10

        if train:
            dataset = dataset(root='../datasets/cifar10/', download=True, train=True, transform=transform)
            return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
        else:
            test_dataset = dataset(root='../datasets/cifar10/', download=True, train=False, transform=transform)
            return DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


def mnist(image_size=32, train=True, batch_size=64) -> DataLoader:
    transform = Compose([Resize(image_size),
                         ToTensor(),
                         # Normalize([0.5], [0.5])
                         ])
    dataset = MNIST
    if train:
        dataset = dataset(root='../datasets/mnist/', download=True, train=True, transform=transform)
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    else:
        test_dataset = dataset(root='../datasets/mnist/', download=True, train=False, transform=transform)
        return DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


def fashion(image_size=32, train=True, batch_size=64) -> DataLoader:
    transform = Compose([Resize(image_size),
                         ToTensor(),
                         # Normalize([0.5], [0.5])
                         ])
    dataset = FashionMNIST
    if train:
        dataset = dataset(root='../datasets/fashionMNIST/', download=True, train=True, transform=transform)
        return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)
    else:
        test_dataset = dataset(root='../datasets/fashionMNIST/', download=True, train=False, transform=transform)
        return DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

