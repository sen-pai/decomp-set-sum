import numpy as np
import pandas as pd 
import os
import glob

from typing import Dict

from .constants import DEPTS, YEARS

def valid_combinations_from_csv(csv_file_name: str, tif_path_origin: str = None, split: int = 64) -> Dict:
    df = pd.read_csv(csv_file_name)

    comb_and_production = dict()
    comb_and_production.fromkeys(DEPTS, [])

    flattened_comb_and_production = dict()

    num_valid = 0
    for dept in DEPTS:
        valid_years = list(df[df.department == dept].year)
        valid_production = [[i] for i in list(df[df.department == dept].production)]

        num_valid += len(valid_years)
        comb_and_production[dept] = dict(zip(valid_years, valid_production))

    count = 0
    for dept in comb_and_production.keys():
        for valid_year in comb_and_production[dept].keys():
            paths = glob.glob(f'{tif_path_origin}/{dept}/{valid_year}/split*_{split}/*.tif')
            comb_and_production[dept][valid_year].append(paths)

            flattened_comb_and_production[count] = comb_and_production[dept][valid_year]
            count += 1
    
    return comb_and_production, flattened_comb_and_production, num_valid





def valid_combinations_from_csv_pkl(csv_file_name: str, tif_path_origin: str = None, depts = DEPTS) -> Dict:
    df = pd.read_csv(csv_file_name)

    comb_and_production = dict()
    comb_and_production.fromkeys(depts, [])

    flattened_comb_and_production = dict()

    num_valid = 0
    for dept in depts:
        valid_years = list(df[df.department == dept].year)
        valid_production = [[i] for i in list(df[df.department == dept].production)]

        num_valid += len(valid_years)
        comb_and_production[dept] = dict(zip(valid_years, valid_production))

    count = 0
    for dept in comb_and_production.keys():
        for valid_year in comb_and_production[dept].keys():
            paths = glob.glob(f'{tif_path_origin}/{dept}/{valid_year}/{dept}_{valid_year}_groups/*.pkl')
            comb_and_production[dept][valid_year].append(paths)

            flattened_comb_and_production[count] = comb_and_production[dept][valid_year]
            count += 1
    
    return comb_and_production, flattened_comb_and_production, num_valid







def valid_combinations_from_csv_small_production(csv_file_name: str, tif_path_origin: str = None, split: int = 64, prod_limit: int = 1000) -> Dict:
    df = pd.read_csv(csv_file_name)

    comb_and_production = dict()
    comb_and_production.fromkeys(DEPTS, [])

    flattened_comb_and_production = dict()

    num_valid = 0
    for dept in DEPTS:
        valid_years = list(df[df.department == dept & df.production >= prod_limit].year)
        valid_production = [[i] for i in list(df[df.department == dept].production)]

        num_valid += len(valid_years)
        comb_and_production[dept] = dict(zip(valid_years, valid_production))

    count = 0
    for dept in comb_and_production.keys():
        for valid_year in comb_and_production[dept].keys():
            paths = glob.glob(f'{tif_path_origin}/{dept}/{valid_year}/split*_{split}/*.tif')
            comb_and_production[dept][valid_year].append(paths)

            flattened_comb_and_production[count] = comb_and_production[dept][valid_year]
            count += 1
    
    return comb_and_production, flattened_comb_and_production, num_valid

