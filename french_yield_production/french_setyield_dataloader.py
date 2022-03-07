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



def check_many_zeros(array: np.array, width: int, height: int) -> bool:
    uni, count = np.unique(array, return_counts=True)

    if count[0] > 0.7 * width * height * 6:
        # print(count[0])
        return True
    return False



class FrenchSetYieldDataset(Dataset):
    def __init__(
        self,
        csv_production_file_name: str,
        tif_path_origin: str,
        normalize: bool = True,
        width: int = 64,
    ):
        """
        Default format is channels first.
        width is expected to be the same as height
        """
        (
            self.paths_and_production_dict,
            self.flat_valid_dict,
            self.num_valid,
        ) = valid_combinations_from_csv(csv_production_file_name, tif_path_origin)

        self.normalize = normalize
        self.width = width

    def __len__(self) -> int:
        return self.num_valid

    def torchify(self, x: np.array) -> torch.tensor:
        x = torch.tensor(x).float()
        if self.normalize:
            return x / 254.0
        return x


    def __getitem__(self, item: int):

        the_sum = self.flat_valid_dict[item][0]
        images = []
        for tif_path in self.flat_valid_dict[item][1]:
            img = load_merged_subtile(tif_path, self.width, self.width)

            if not check_many_zeros(img, self.width, self.width):
                img = self.torchify(img)
                images.append(img)

        return (
            torch.stack(images, dim=0),
            torch.tensor(the_sum),
        )




class FrenchLSTMSetYieldDataset(Dataset):
    def __init__(
        self,
        csv_production_file_name: str,
        tif_path_origin: str,
        normalize: bool = True,
        width: int = 64,
    ):
        """
        Default format is channels first.
        width is expected to be the same as height
        """
        (
            self.paths_and_production_dict,
            self.flat_valid_dict,
            self.num_valid,
        ) = valid_combinations_from_csv(csv_production_file_name, tif_path_origin, width)

        self.normalize = normalize
        self.width = width
        self.good_items = []

    def __len__(self) -> int:
        return self.num_valid

    def torchify(self, x: np.array) -> torch.tensor:
        x = torch.tensor(x).float()
        if self.normalize:
            x =  x / 254.0
        x = torch.flatten(x, 1)
        return x


    def __getitem__(self, item: int):

        the_sum = self.flat_valid_dict[item][0]
        images = []
        for tif_path in self.flat_valid_dict[item][1]:
            img = load_merged_subtile(tif_path, self.width, self.width)
            if not check_many_zeros(img, self.width, self.width):
                img = self.torchify(img)
                images.append(img)

            # img = self.torchify(img)
            # images.append(img)

        
        if images:
            self.good_items.append(item)
        else:
            item = random.choice(self.good_items)
            the_sum = self.flat_valid_dict[item][0]
            images = []
            for tif_path in self.flat_valid_dict[item][1]:
                img = load_merged_subtile(tif_path, self.width, self.width)
                if not check_many_zeros(img, self.width, self.width):
                    img = self.torchify(img)
                    images.append(img)

        
        return (
            torch.stack(images, dim=0),
            torch.tensor(the_sum),
        )


if __name__ == "__main__":
    from utils.valid_combinations import valid_combinations_from_csv

    main_dict, flat_dict, num = valid_combinations_from_csv(
        "./winter_wheat_filtered_2002.csv", "../french_dept_data"
    )

    shapes = FrenchSetYieldDataset("./winter_wheat_filtered_2002.csv", "../french_dept_data")
    stack, su = shapes.__getitem__(0)
    print(stack.shape)
    print(su)