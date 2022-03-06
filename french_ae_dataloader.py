import random

random.seed(1)

import glob
import os
import itertools
from typing import List

import numpy as np


import rasterio

import torch
from torch.utils.data import Dataset


def nodata_to_zero(array: np.array, no_data: int) -> np.array:
    array = np.where(array != no_data, array, 0)
    return array


def load_merged_subtile(file_name: str, width: int, height: int) -> np.array:
    # Channels, H, W
    subtile = np.zeros((6, width, height))

    with rasterio.open(file_name) as src:
        temp_arr = src.read()
        no_data = src.nodata

    if no_data != 0:
        temp_arr = nodata_to_zero(temp_arr, no_data)

    subtile[:, : temp_arr.shape[1], : temp_arr.shape[2]] = temp_arr

    return subtile




class FrenchAEDataset(Dataset):
    def __init__(
        self,
        dataset_len: int,
    ):
        """
        Return a tuple of two random (x, y) and 1 if x = y, or -1 if x != y
        """
        self.dataset_len = dataset_len

    def __len__(self) -> int:
        return self.dataset_len

    def torchify(self, x: np.array, norm: bool = True) -> torch.tensor:
        x = torch.tensor(x)
        if norm:
            return x / 255.0
        return x

    def __getitem__(self, item: int) -> Tuple[FloatTensor, FloatTensor, FloatTensor]:

        x_img, x_target = self.shape_funcs[random.randint(0, 3)]()
        x_img = self.torchify(x_img)

        if random.uniform(0, 1) >= self.same_prob:
            y_img, y_target = self.shape_funcs[random.randint(0, 3)]()
            y_img = self.torchify(y_img)
        else:
            y_img = copy.deepcopy(x_img)
            y_target = copy.deepcopy(x_target)

        indicator = 1 if x_target == y_target else -1

        return x_img, y_img, indicator