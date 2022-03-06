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
from utils.valid_combinations import valid_combinations_from_csv


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


class FrenchSetYieldDataset(Dataset):
    def __init__(self, csv_production_file_name: str, tif_path_origin: str, normalize: bool = True, width:int = 64):
        """
        Default format is channels first.
        width is expected to be the same as height
        """
        self.paths_and_production_dict = valid_combinations_from_csv(csv_production_file_name, tif_path_origin)
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



if __name__ == "__main__":
    from utils.valid_combinations import valid_combinations_from_csv

    print(valid_combinations_from_csv('./winter_wheat_filtered_2002.csv', "../french_dept_data")['Ain'])