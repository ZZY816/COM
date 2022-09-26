import argparse
import glob
import os.path
from pathlib import Path
import pickle

import numpy as np
import torch


from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils

root = '/media/didi/1.0TB/waymo'
file = 'waymo_processed_data_v0_5_0_waymo_dbinfos_train_sampled_1.pkl'
info_path = os.path.join(root, file)
with open(info_path, 'rb') as f:
    db_infos = pickle.load(f)

car_info = db_infos['Vehicle']
npgt = np.array([car_info[i]['num_points_in_gt'] for i in range(len(car_info))])
distance = np.array([np.sqrt(np.power(car_info[i]['box3d_lidar'][0], 2) + np.power(car_info[i]['box3d_lidar'][1], 2))
                     for i in range(len(car_info))])

print(distance.min(), distance.max())


out_file = 'dbinfos_car_by_num_points_in_gt_distance.pkl'
out_dict = {}
out_dict['60'] = []
out_dict['50_60'] = []
out_dict['40_50'] = []
out_dict['30_40'] = []
out_dict['20_30'] = []
out_dict['10_20'] = []
out_dict['0_10'] = []

for i in np.random.permutation(len(car_info)):
    if i%1000 ==0:
        print(i)
    if distance[i] <= 10 and len(out_dict['0_10']) <= 1000:
        out_dict['0_10'].append(car_info[i])
    elif 10 < distance[i] <= 20 and len(out_dict['10_20']) <= 1000:
        out_dict['10_20'].append(car_info[i])
    elif 20 < distance[i] <= 30 and len(out_dict['20_30']) <= 1000:
        out_dict['20_30'].append(car_info[i])
    elif 30 < distance[i] <= 40 and len(out_dict['30_40']) <= 1000:
        out_dict['30_40'].append(car_info[i])
    elif 40 < distance[i] <= 50 and len(out_dict['40_50']) <= 1000:
        out_dict['40_50'].append(car_info[i])
    elif 50 < distance[i] <= 60 and len(out_dict['50_60']) <= 1000:
        out_dict['50_60'].append(car_info[i])
    elif 60 < distance[i] and len(out_dict['60']) <= 1000:
        out_dict['60'].append(car_info[i])
    else:
        continue



# x = car_info[0]
# print(x)
with open(os.path.join(root, os.path.join(root, out_file)), 'wb') as f1:
    pickle.dump(out_dict, f1)
