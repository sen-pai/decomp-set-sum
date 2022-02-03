# taken from https://github.com/yassersouri/pytorch-deep-sets/blob/master/src/deepsets/datasets.py


from typing import Tuple

import numpy as np
import torch
from torch import FloatTensor
from torch.utils.data.dataset import Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize

DATA_ROOT = "./"

MNIST_TRANSFORM = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])


class MNISTSummation(Dataset):
    def __init__(self, min_len: int, max_len: int, dataset_len: int, train: bool = True, transform: Compose = MNIST_TRANSFORM):
        self.min_len = min_len
        self.max_len = max_len
        self.dataset_len = dataset_len
        self.train = train
        self.transform = transform

        self.mnist = MNIST(DATA_ROOT, train=self.train, transform=self.transform, download=True)
        mnist_len = self.mnist.__len__()
        mnist_items_range = np.arange(0, mnist_len)

        items_len_range = np.arange(self.min_len, self.max_len + 1)
        items_len = np.random.choice(items_len_range, size=self.dataset_len, replace=True)
        self.mnist_items = []
        for i in range(self.dataset_len):
            self.mnist_items.append(np.random.choice(mnist_items_range, size=items_len[i], replace=True))

    def __len__(self) -> int:
        return self.dataset_len

    def __getitem__(self, item: int) -> Tuple[FloatTensor, FloatTensor]:
        mnist_items = self.mnist_items[item]

        the_sum = 0
        images = []
        targets = []
        for mi in mnist_items:
            img, target = self.mnist.__getitem__(mi)
            the_sum += target
            images.append(img)
            targets.append(target)

        return torch.stack(images, dim=0), torch.tensor(targets), torch.FloatTensor([the_sum])



if __name__ == "__main__":
    train_db = MNISTSummation(min_len=2, max_len=10, dataset_len=100000, train=True, transform=MNIST_TRANSFORM)
    print(train_db.__getitem__(0))