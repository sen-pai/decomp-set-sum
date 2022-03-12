import pandas as pd 
import rasterio as rio
import numpy as np

import glob

from joblib import Parallel, delayed


from utils.constants import DEPTS, YEARS


from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

import shutil
import os

import pickle


dataset_path = os.path.join(os.getcwd(), "../french_dept_data")




def nodata_to_zero(array: np.array, no_data: int) -> np.array:
    array = np.where(array != no_data, array, 0)
    return array


def load_merged_subtile(file_name: str, width: int, height: int) -> np.array:
    # Channels, H, W
    subtile = np.zeros((6, width, height))

    with rio.open(file_name) as src:
        temp_arr = src.read()
        no_data = src.nodata

    if no_data != 0:
        temp_arr = nodata_to_zero(temp_arr, no_data)

    subtile[:, : temp_arr.shape[1], : temp_arr.shape[2]] = temp_arr

    return subtile


def load_pixel(file_name: str) -> np.array:
    with rio.open(file_name) as src:
        temp_arr = src.read()
        no_data = src.nodata

    if no_data != 0:
        temp_arr = nodata_to_zero(temp_arr, no_data)
    
    return temp_arr

def check_zero(number):
    if np.all(number == 0):
        return False  
    return True


def copy_paste_files(paths, dest_folder):
    if os.path.exists(dest_folder):
        shutil.rmtree(dest_folder)
    os.mkdir(dest_folder)

    for path in paths:
        shutil.copy(path, dest_folder) 


def flatten(t):
    return [item for sublist in t for item in sublist]


def get_glob_paths(dept, year):
    return glob.glob(f"../french_dept_data/{dept}/{year}/split*_1/*")


def non_zero_pixels_and_paths(dept, year):
    paths = Parallel(n_jobs=8)(delayed(get_glob_paths)(d, y) for y in [year] for d in [dept])
    paths = flatten(paths)
    print(f'{dept} for {year} has pixels = {len(paths)}')

    non_empty_pixels = Parallel(n_jobs=8)(delayed(load_pixel)(path) for path in paths)

    non_zero_paths = []
    non_empty_pixels_ = []

    for pixel, path in zip(non_empty_pixels, paths):
        if check_zero(pixel):
            non_empty_pixels_.append(pixel)
            non_zero_paths.append(path)

    print(f'{dept} for {year} has non empty pixels = {len(non_zero_paths)}')
    

    return non_empty_pixels_, non_zero_paths

def to_groups(pixels, paths):
    pixel_groups = dict()

    count = 0
    while len(paths) > 1:
        count += 1
        pixel_group = []
        pixel_paths_group = []
        pixel_group_indices = []
        pixel_a = pixels[0]
        for index, (pixel, path) in enumerate(zip(pixels, paths)):
            if index == 0:
                pixel_group.append(pixel)
                pixel_paths_group.append(path)
                pixel_group_indices.append(index)
            else:
                if pearsonr(pixel_a.reshape(-1), pixel.reshape(-1))[0] > 0.9:
                    if cdist(pixel_a.reshape(1,-1), pixel.reshape(1,-1))[0] < 50:
                        pixel_group.append(pixel)
                        pixel_paths_group.append(path)
                        pixel_group_indices.append(index)
                
        paths = [j for i, j in enumerate(paths) if i not in pixel_group_indices]
        pixels = [j for i, j in enumerate(pixels) if i not in pixel_group_indices]

        # print(len(pixel_paths_group))
        
        pixel_groups[count] = [pixel_group, pixel_paths_group]

    print(f' Found {len(pixel_groups.keys())} groups ')
    
    return pixel_groups




for dept in DEPTS:
    for year in YEARS:
        dept_year_path = os.path.join(dataset_path, dept, year)
        output_folder_name = f"{dept}_{year}_groups"
        output_folder_path = os.path.join(dept_year_path, output_folder_name)

        if os.path.exists(output_folder_path):
            shutil.rmtree(output_folder_path)
        os.mkdir(output_folder_path)

        output_filename = f'{dept}_{year}_groups.pkl'

        non_zero_pixels, non_zero_paths = non_zero_pixels_and_paths(dept, year)

        groups_dict = to_groups(non_zero_pixels, non_zero_paths)

        with open(os.path.join(output_folder_path, output_filename), 'wb') as f:
            pickle.dump(groups_dict, f)