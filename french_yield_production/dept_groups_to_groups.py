import pandas as pd 
import rasterio as rio
import numpy as np

import glob
import copy


from utils.constants import DEPTS, YEARS


from scipy.spatial.distance import cdist
from scipy.stats import pearsonr

import shutil
import os

import pickle


dataset_path = os.path.join(os.getcwd(), "../french_dept_data")


# def merge_two_areas(area_a, area_b):

#     super_group = dict()

#     merged_groups_b = []
#     merged_groups_a = []
#     count = 1
#     for i, group_a in enumerate(area_a):
#         if i not in merged_groups_a:
#             for j, group_b in enumerate(area_b):
#                 if j not in merged_groups_b:
#                     # print(group_a, group_b)
#                     if pearsonr(area_a[group_a][0][0].reshape(-1), area_b[group_b][0][0].reshape(-1))[0] > 0.9:
#                         if cdist(area_a[group_a][0][0].reshape(1,-1), area_b[group_b][0][0].reshape(1,-1))[0] < 50:

#                             list_pixels = copy.deepcopy(area_a[group_a][0])
#                             list_pixels.extend(area_b[group_b][0])

#                             list_filenames = copy.deepcopy(area_a[group_a][1])
#                             list_filenames.extend(area_b[group_b][1])

#                             super_group[count] = [list_pixels, list_filenames]
                            
#                             count += 1
#                             merged_groups_a.append(i)
#                             merged_groups_b.append(j)

#     print(f'These many common sub groups found: {len(merged_groups_a)}')

#     for i, group_a in enumerate(area_a):
#         if i not in merged_groups_a:
#             super_group[count] = area_a[group_a]
#             count += 1

#     for i, group_b in enumerate(area_b):
#         if i not in merged_groups_b:
#             super_group[count] = area_b[group_b]
#             count += 1

#     print(len(super_group.keys()))

#     return super_group


def merge_two_areas(area_a, area_b):

    super_group = dict()

    merged_groups_b = []
    merged_groups_a = []
    count = 1

    for i, group_a in enumerate(area_a):
        sim_groups = []
        for j, group_b in enumerate(area_b):
            if j not in merged_groups_b:
                if pearsonr(area_a[group_a][0][0].reshape(-1), area_b[group_b][0][0].reshape(-1))[0] > 0.9:
                    if cdist(area_a[group_a][0][0].reshape(1,-1), area_b[group_b][0][0].reshape(1,-1))[0] < 50:
                        
                        merged_groups_b.append(j)
                        sim_groups.append(group_b)
                        # print(sim_groups)

        list_pixels = copy.deepcopy(area_a[group_a][0])
        list_filenames = copy.deepcopy(area_a[group_a][1])

        for sim_key in sim_groups:
            list_pixels.extend(area_b[sim_key][0])
            list_filenames.extend(area_b[sim_key][1])

        super_group[count] = [list_pixels, list_filenames]

        count += 1

    print(f'These many common sub groups found: {len(merged_groups_b)}')

    # for i, group_a in enumerate(area_a):
    #     if i not in merged_groups_a:
    #         super_group[count] = area_a[group_a]
    #         count += 1

    for i, group_b in enumerate(area_b):
        if i not in merged_groups_b:
            super_group[count] = area_b[group_b]
            count += 1

    print(len(super_group.keys()))

    return super_group


depts = ['Aisne', 'Allier', 'Ardennes', 'Aube', 'Bas_Rhin', 'Val_de_Marne', 'Var', 'Vaucluse', 'Vosges', 'Alpes_de_Haute_Provence', 'Alpes_Maritimes']


dept_areas = []
super_group_dept = []
for dept in depts:
    dept_path = os.path.join(dataset_path, dept)
    output_filename = f'{dept}_all_groups_merged.pkl'
    dept_areas.append(os.path.join(dept_path, output_filename))
    

with open(dept_areas[0], 'rb') as f:
    super_group_dept = pickle.load(f)
    
for i, areas in enumerate(dept_areas):
    if i != 0:
        with open(dept_areas[i], 'rb') as f:
            to_merge_area = pickle.load(f)
        
        super_group_dept = merge_two_areas(super_group_dept, to_merge_area)
    
    dept_merged_name = f'subset_dept_groups_merged.pkl'
    with open(os.path.join(dataset_path, dept_merged_name), 'wb') as f:
            pickle.dump(super_group_dept, f)

    print(f'till area {areas} saved')

