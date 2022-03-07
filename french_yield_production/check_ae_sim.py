
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

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch.optim as optim

from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from french_ae_dataloader import FrenchLSTMAEDatasetCheck
from french_lstm_model import RNNDecoder, RNNEncoder, Seq2SeqAttn

from utils.constants import TrainDEPTS, TrainYEARS, ValDEPTS, ValYEARS


from sklearn.metrics.pairwise import cosine_similarity

width_dim = 32


train_subtile_paths = []
for dept in TrainDEPTS:
    for year in TrainYEARS:
        train_subtile_paths.extend(glob.glob(f"../french_dept_data/{dept}/{year}/split*_{width_dim}/*"))

train_dataset = FrenchLSTMAEDatasetCheck(train_subtile_paths, normalize= False, width = width_dim)



val_subtile_paths = []
for dept in ValDEPTS:
    for year in ValYEARS:
        val_subtile_paths.extend(glob.glob(f"../french_dept_data/{dept}/{year}/split*_{width_dim}/*"))

valid_dataset = FrenchLSTMAEDatasetCheck(val_subtile_paths, normalize= False, width = width_dim)

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




def most_similar(input_tif, model):
    
    input_tif = input_tif.to(device)
    _, h_main, _ = model.encoder.encode(input_tif.unsqueeze(0), torch.tensor([6]))

    h_main = h_main.view(1,-1).detach().cpu().numpy()

    for input_targets, file_name in tqdm(dataloaders['train']):
        
        input_targets = input_targets.to(device)

        _, h, _ = model.encoder.encode(input_targets, torch.tensor([6]))
        h = h.view(1,-1).detach().cpu().numpy()

        # print(cosine_similarity(h_main, h))
        # print(file_name)
        # break

        if cosine_similarity(h_main, h) > 0.95:
            print(file_name)



device = torch.device('cuda')


e = RNNEncoder(input_dim=width_dim*width_dim, bidirectional=True)
d = RNNDecoder(
    input_dim=(e.input_size + e.hidden_size * 2),
    hidden_size=e.hidden_size,
    bidirectional=True,
    output_dim= e.input_size,
)

model = Seq2SeqAttn(encoder=e, decoder=d).to(device)
model.load_state_dict(torch.load('experiments/ae_32_lstm/weights/ae_32_lstm_weights_60.pt'))
model.eval()  # Set model to evaluate mode

input_tif, file_n = train_dataset.__getitem__(0, '../french_dept_data/Aisne/2002/split_Aisne_2002_32/subtile_32-32.tif')

print("main file", file_n)
most_similar(input_tif, model)
