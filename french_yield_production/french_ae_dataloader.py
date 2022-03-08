import random

random.seed(1)

import glob
import os
import itertools
from typing import Callable, List

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

def check_many_zeros(array: np.array, width: int, height: int) -> bool:
    uni, count = np.unique(array, return_counts=True)

    if count[0] > 0.5 * width * height * 6:
        # print(count[0])
        return True
    return False

def get_histogram(occurances: np.array, drop_zero: bool = True, normalize: bool = False):
    # print("uni", np.unique(occurances))
    uni, count = np.unique(occurances, return_counts=True)
    zero_num = count[0] if uni[0] == 0 else 0


    occurances = occurances.astype(int)
    occurances = occurances.flatten()
    total = occurances.shape[0]
    # print(occurances.shape)
    hist = np.bincount(occurances, minlength= 255)

    if normalize:
        hist = hist/(total - zero_num + 1e-6)
    # print(hist.shape)
    if drop_zero:
        # print("reached")
        hist = np.delete(hist, [0])
    
    return hist

class FrenchAEDataset(Dataset):
    def __init__(self, file_paths: List, normalize: bool = True, width:int = 64):
        """
        Default format is channels first.
        width is expected to be the same as height
        """
        self.file_paths = file_paths
        self.normalize = normalize
        self.width = width
        self.good_indices = [4]

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

        if check_many_zeros(subtile, self.width, self.width):
            file_name = self.file_paths[random.choice(self.good_indices)]
        else:
            self.good_indices.append(index)

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
        self.good_indices = [4]

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

        if check_many_zeros(subtile, self.width, self.width):
            file_name = self.file_paths[random.choice(self.good_indices)]
        else:
            self.good_indices.append(index)

        subtile = self.torchify(subtile)

        return subtile, subtile


class FrenchLSTMAEDatasetCheck(Dataset):
    def __init__(self, file_paths: List, normalize: bool = True, width:int = 64):
        """
        Default format is channels first.
        width is expected to be the same as height
        """
        self.file_paths = file_paths
        self.normalize = normalize
        self.width = width
        self.good_indices = [4]

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

    def __getitem__(self, index: int, file_name = None):
        if not file_name:
            file_name = self.file_paths[index]

        subtile = load_merged_subtile(file_name, self.width, self.width)

        if check_many_zeros(subtile, self.width, self.width):
            file_name = self.file_paths[random.choice(self.good_indices)]
        else:
            self.good_indices.append(index)


        subtile = self.torchify(subtile)

        return subtile, file_name




class FrenchHistDataset(Dataset):
    def __init__(self, file_paths: List, normalize: bool = False, width:int = 64):
        """
        Default format is channels first.
        width is expected to be the same as height
        """
        self.file_paths = file_paths
        self.normalize = normalize
        self.width = width
        self.good_indices = [4]

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, index: int, file_name = None):
        if not file_name:
            file_name = self.file_paths[index]

        subtile = load_merged_subtile(file_name, self.width, self.width)
        bincount = get_histogram(subtile)

        if check_many_zeros(subtile, self.width, self.width):
            file_name = self.file_paths[random.choice(self.good_indices)]
        else:
            self.good_indices.append(index)

        subtile = self.torchify(subtile)

        return subtile, bincount, file_name



class FrenchSimHistDataset(Dataset):
    def __init__(self, file_paths: List, normalize: bool = False, width:int = 64, similarity_func: Callable = None):
        """
        Default format is channels first.
        width is expected to be the same as height
        """
        self.file_paths = file_paths
        self.normalize = normalize
        self.width = width
        self.good_indices = [4]
        self.similarity_func = similarity_func

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

    def __getitem__(self, index: int, file_name = None):
        if not file_name:
            file_name = self.file_paths[index]

        subtile_a = load_merged_subtile(file_name, self.width, self.width)
        bincount_a = get_histogram(subtile_a)

        if check_many_zeros(subtile_a, self.width, self.width):
            file_name = self.file_paths[random.choice(self.good_indices)]
        else:
            self.good_indices.append(index)

        subtile_a = self.torchify(subtile_a)


        subtile_b = load_merged_subtile(random.choice(self.file_paths), self.width, self.width)
        bincount_b = get_histogram(subtile_b)

        if check_many_zeros(subtile_b, self.width, self.width):
            file_name = self.file_paths[random.choice(self.good_indices)]
        else:
            self.good_indices.append(index)

        subtile_b = self.torchify(subtile_b)

        indicator = self.similarity_func(bincount_a, bincount_b)
        
        return subtile_a, subtile_b, indicator


if __name__ == "__main__":
    paths = []
    for dept in DEPTS:
        for year in YEARS:
            paths.extend(glob.glob(f"../french_dept_data/{dept}/{year}/split*_64/*"))

    a = FrenchAEDataset(paths)
    print(a.__len__())

    # print(paths)