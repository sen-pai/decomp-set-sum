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


from utils.constants import DEPTS, YEARS


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
    def __init__(self, file_paths: List, normalize: bool = True, width:int = 64):
        """
        Default format is channels first.
        width is expected to be the same as height
        """
        self.file_paths = file_paths
        self.normalize = normalize
        self.width = width

    def __len__(self) -> int:
        return len(self.file_paths)

    def torchify(self, x: np.array) -> torch.tensor:
        x = torch.tensor(x).float()
        if self.normalize:
            return x / 254.0
        return x

    def __getitem__(self, index: int):
        file_name = self.file_paths[index]

        subtile = load_merged_subtile(file_name, self.width, self.width)
        subtile = self.torchify(subtile)

        return subtile, subtile



class FrenchLSTMAEDataset(Dataset):
    def __init__(self, file_paths: List, normalize: bool = True, width:int = 64):
        """
        Default format is channels first.
        width is expected to be the same as height
        """
        self.file_paths = file_paths
        self.normalize = normalize
        self.width = width

    def __len__(self) -> int:
        return len(self.file_paths)

    def torchify(self, x: np.array) -> torch.tensor:
        x = torch.tensor(x).float()
        if self.normalize:
            x =  x / 254.0

        # shape (6, 64, 64) to (6, 4096)
        x = torch.flatten(x, 1)
        # print("x", x.shape)
        return x

    def __getitem__(self, index: int):
        file_name = self.file_paths[index]

        subtile = load_merged_subtile(file_name, self.width, self.width)
        subtile = self.torchify(subtile)

        return subtile, subtile



if __name__ == "__main__":
    paths = []
    for dept in DEPTS:
        for year in YEARS:
            paths.extend(glob.glob(f"../french_dept_data/{dept}/{year}/split*_64/*"))

    a = FrenchAEDataset(paths)
    print(a.__len__())

    # print(paths)