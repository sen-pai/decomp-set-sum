import numpy as np
import pandas as pd 
import os
import glob

from typing import Dict

from .constants import DEPTS, YEARS

def valid_combinations_from_csv(csv_file_name: str, tif_path_origin: str = None) -> Dict:
    df = pd.read_csv(csv_file_name)

    comb_and_production = dict()
    comb_and_production.fromkeys(DEPTS, [])

    flattened_comb_and_production = dict()

    num_valid = 0
    for dept in DEPTS:
        valid_years = list(df[df.department == "Ain"].year)
        valid_production = [[i] for i in list(df[df.department == "Ain"].production)]

        num_valid += len(valid_years)
        comb_and_production[dept] = dict(zip(valid_years, valid_production))

    count = 0
    for dept in comb_and_production.keys():
        for valid_year in comb_and_production[dept].keys():
            paths = glob.glob(f'{tif_path_origin}/{dept}/{valid_year}/split*_64/*.tif')
            comb_and_production[dept][valid_year].append(paths)

            flattened_comb_and_production[count] = comb_and_production[dept][valid_year]
            count += 1
    
    return comb_and_production, flattened_comb_and_production, num_valid



