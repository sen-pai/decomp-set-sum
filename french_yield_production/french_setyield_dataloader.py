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

import pickle
from utils.constants import DEPTS, YEARS
from utils.valid_combinations import valid_combinations_from_csv, valid_combinations_from_csv_pkl



from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

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
        ) = valid_combinations_from_csv(csv_production_file_name, tif_path_origin, split=  width)

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




class FrenchPickleDataset(Dataset):
    def __init__(
        self,
        csv_production_file_name: str,
        tif_path_origin: str,
        depts: List = DEPTS,
        normalize: bool = True,
        norm_sum: int = 1
    ):
        """
        Default format is channels first.
        width is expected to be the same as height
        """
        (
            self.paths_and_production_dict,
            self.flat_valid_dict,
            self.num_valid,
        ) = valid_combinations_from_csv_pkl(csv_production_file_name, tif_path_origin, depts)

        self.normalize = normalize
        self.norm_sum = norm_sum

    def __len__(self) -> int:
        return self.num_valid

    def torchify(self, x: np.array) -> torch.tensor:
        x = torch.tensor(x).float()
        if self.normalize:
            x =  x / 254.0
        x = torch.flatten(x, 1)
        return x


    def __getitem__(self, item: int):

        the_sum = self.flat_valid_dict[item][0] / self.norm_sum

        the_sum = np.around(the_sum,3)
        # the_sum /= self.norm_sum
        # the_sum = round(the_sum, 3)
        # print(the_sum)
        
        pixels = []
        group_sizes = []

        # print(self.flat_valid_dict[item][1])
        with open(self.flat_valid_dict[item][1][0], 'rb') as f:
            pkl_groups = pickle.load(f)

        
        for np_pixels, paths in pkl_groups.values():
            group_sizes.append(len(paths))
            # pixels.append(self.torchify(random.sample(np_pixels, 1)[0]))
            pixels.append(self.torchify(np_pixels[0]))
            
            
        
        return (
            torch.stack(pixels, dim=0),
            torch.tensor(group_sizes),
            torch.tensor(the_sum)
        )



class FrenchLinearPickleDataset(Dataset):
    def __init__(
        self,
        csv_production_file_name: str,
        tif_path_origin: str,
        depts: List = DEPTS,
        normalize: bool = True,
        norm_sum: int = 1
    ):
        """
        Default format is channels first.
        width is expected to be the same as height
        """
        (
            self.paths_and_production_dict,
            self.flat_valid_dict,
            self.num_valid,
        ) = valid_combinations_from_csv_pkl(csv_production_file_name, tif_path_origin, depts)

        self.normalize = normalize
        self.norm_sum = norm_sum

    def __len__(self) -> int:
        return self.num_valid

    def torchify(self, x: np.array) -> torch.tensor:
        x = torch.tensor(x).float()
        if self.normalize:
            x =  x / 254.0
        x = torch.flatten(x, 1)
        return x


    def __getitem__(self, item: int):

        the_sum = self.flat_valid_dict[item][0] / self.norm_sum

        the_sum = np.around(the_sum,3)
        # the_sum /= self.norm_sum
        # the_sum = round(the_sum, 3)
        # print(the_sum)
        
        pixels = []
        group_sizes = []

        # print(self.flat_valid_dict[item][1])
        with open(self.flat_valid_dict[item][1][0], 'rb') as f:
            pkl_groups = pickle.load(f)

        
        for np_pixels, paths in pkl_groups.values():
            group_sizes.append(len(paths))
            # pixels.append(self.torchify(random.sample(np_pixels, 1)[0]))
            pixels.append(self.torchify(np_pixels[0]))
            
            
        
        return (
            torch.stack(pixels, dim=0).view(-1, 6).float(),
            torch.tensor(group_sizes),
            torch.tensor(the_sum).float()
        )



class FrenchSimLinearPickleDataset(Dataset):
    def __init__(
        self,
        csv_production_file_name: str,
        tif_path_origin: str,
        depts: List = DEPTS,
        normalize: bool = True,
        norm_sum: int = 1
    ):
        """
        Default format is channels first.
        width is expected to be the same as height
        """
        (
            self.paths_and_production_dict,
            self.flat_valid_dict,
            self.num_valid,
        ) = valid_combinations_from_csv_pkl(csv_production_file_name, tif_path_origin, depts)

        self.normalize = normalize
        self.norm_sum = norm_sum

    def __len__(self) -> int:
        return self.num_valid

    def torchify(self, x: np.array) -> torch.tensor:
        x = torch.tensor(x).float()
        if self.normalize:
            x =  x / 254.0
        x = torch.flatten(x, 1)
        return x

    def similarity_func(self, a, b):
        if pearsonr(a.reshape(-1), b.reshape(-1))[0] > 0.9:
            if cdist(a.reshape(1,-1), b.reshape(1,-1))[0] < 50:
                return 1
        return -1

    def __getitem__(self, item: int):

        # print(self.flat_valid_dict[item][1])
        with open(self.flat_valid_dict[item][1][0], 'rb') as f:
            pkl_groups_a = pickle.load(f)

        random_key = random.sample(list(range(self.num_valid)), 1)[0]
        with open(self.flat_valid_dict[random_key][1][0], 'rb') as f:
            pkl_groups_b = pickle.load(f)

        a_key = random.sample(pkl_groups_a.keys(), 1)[0]
        a = pkl_groups_a[a_key][0][0]

        if random.random() > 0.5:
            # pixels_a = pkl_groups_a[a_key][0]
            b = pkl_groups_a[a_key][0][0]
            # print(b)

        else:
            b = pkl_groups_b[random.sample(pkl_groups_b.keys(), 1)[0]][0][0]
        
        # print(a)
        # print(b)

        sim = self.similarity_func(a, b)

        a = self.torchify(a)
        b = self.torchify(b)
        
        return (
            a.view(-1).float().cpu(),
            b.view(-1).float().cpu(),
            torch.tensor(sim).float().cpu()
        )


if __name__ == "__main__":
    from utils.valid_combinations import valid_combinations_from_csv

    main_dict, flat_dict, num = valid_combinations_from_csv(
        "./winter_wheat_filtered_2002.csv", "../french_dept_data", 64
    )

    # print(main_dict)

    # shapes = FrenchSetYieldDataset("./winter_wheat_filtered_2002.csv", "../french_dept_data")
    # stack, su = shapes.__getitem__(0)
    # print(stack.shape)
    # print(su)

    fpkl = FrenchPickleDataset("./winter_wheat_filtered_2002.csv", "../french_dept_data", ['Ain'], norm_sum = 100000)

    p, c, s = fpkl.__getitem__(0)
    print(p.shape)
    print(c.shape)
    print(s)


    fpkl = FrenchSimLinearPickleDataset("./winter_wheat_filtered_2002.csv", "../french_dept_data", ['Ain'], norm_sum = 100000)

    p, c, s = fpkl.__getitem__(0)
    print(p.shape)
    print(c.shape)
    print(s)

    # print((p.*c).shape)