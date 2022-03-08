
import os
import numpy as np
import random
import glob
import copy
import shutil
from collections import defaultdict


import argparse
from tqdm import tqdm
import time

from torch.utils.data import DataLoader

from french_ae_dataloader import FrenchHistDataset

from utils.constants import TrainDEPTS, TrainYEARS, ValDEPTS, ValYEARS


from sklearn.metrics.pairwise import cosine_similarity

from scipy.stats import wasserstein_distance

width_dim = 16


train_subtile_paths = []
for dept in TrainDEPTS:
    for year in TrainYEARS:
        train_subtile_paths.extend(glob.glob(f"../french_dept_data/{dept}/{year}/split*_{width_dim}/*"))

train_dataset = FrenchHistDataset(train_subtile_paths, normalize= False, width = width_dim)



val_subtile_paths = []
for dept in ValDEPTS:
    for year in ValYEARS:
        val_subtile_paths.extend(glob.glob(f"../french_dept_data/{dept}/{year}/split*_{width_dim}/*"))

valid_dataset = FrenchHistDataset(val_subtile_paths, normalize= False, width = width_dim)

dataloaders = {
    "train": DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    ),
    "val": DataLoader(
        valid_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last= True
    ),
}




def most_similar(input_tif):
    print(input_tif)
    # input_tif = np.reshape(input_tif, (1, -1))
    count = 0
    names = []
    for input_targets, file_name in tqdm(dataloaders['train']):

        print("cos sim", cosine_similarity(np.reshape(input_tif, (1, -1)), input_targets))
        # print(input_tif.shape)
        # print(input_targets.shape)

        if input_targets.shape[1] == 253:
            break
        print("Wass ", wasserstein_distance(input_tif, input_targets.view(-1)))
        # print(file_name)
        # brea

        if wasserstein_distance(input_tif, input_targets.view(-1)) < 1.5:
            # print(file_name)
            names.append(file_name)
            count += 1 

    for i in names:
        print(i)

    print(count)
    print(len(set(names)))


input_tif, file_n = train_dataset.__getitem__(0, '../french_dept_data/Aisne/2002/split_Aisne_2002_16/subtile_32-16.tif')

print("main file", file_n)
most_similar(input_tif)
