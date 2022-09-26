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
volume = np.array([car_info[i]['box3d_lidar'][3] * car_info[i]['box3d_lidar'][4] * car_info[i]['box3d_lidar'][5]
                     for i in range(len(car_info))])

distance = np.array([np.sqrt(np.power(car_info[i]['box3d_lidar'][0], 2) + np.power(car_info[i]['box3d_lidar'][1], 2))
                     for i in range(len(car_info))])

density = (npgt/volume) * distance
print(density.max())

density = density/10
# d = density
# x0 = len(d[d < 1])
# x1 = len(d[(d > 1) & (d <= 2)])
# x2 = len(d[(d > 2) & (d <= 4)])
# x3 = len(d[(d > 4) & (d <= 8)])
# x4 = len(d[(d > 8) & (d <= 16)])
# x5 = len(d[(d > 16) & (d <= 32)])
# x6 = len(d[(d > 32) & (d <= 64)])
# x7 = len(d[(d > 64) & (d <= 128)])
# x8 = len(d[d > 128])
#
#
#
# print(x0, x1, x2, x3, x4, x5, x6, x7, x8)







out_file = 'dbinfos_car_by_num_points_in_gt_distance_density.pkl'
out_dict = {}
out_dict['128'] = []
out_dict['64_128'] = []
out_dict['32_64'] = []
out_dict['16_32'] = []
out_dict['8_16'] = []
out_dict['4_8'] = []
out_dict['2_4'] = []
out_dict['1_2'] = []
out_dict['0_1'] = []

for i in np.random.permutation(len(car_info)):
    if i % 1000 ==0:
        print(i)
    if 0 < density[i] <= 1 and len(out_dict['0_1']) <= 1000:
        out_dict['0_1'].append(car_info[i])
    elif 1 < density[i] <= 2 and len(out_dict['1_2']) <= 1000:
        out_dict['1_2'].append(car_info[i])
    elif 2 < density[i] <= 4 and len(out_dict['2_4']) <= 1000:
        out_dict['2_4'].append(car_info[i])
    elif 4 < density[i] <= 8 and len(out_dict['4_8']) <= 1000:
        out_dict['4_8'].append(car_info[i])
    elif 8 < density[i] <= 16 and len(out_dict['8_16']) <= 1000:
        out_dict['8_16'].append(car_info[i])
    elif 16 < density[i] <= 32 and len(out_dict['16_32']) <= 1000:
        out_dict['16_32'].append(car_info[i])
    elif 32 < density[i] <= 64 and len(out_dict['32_64']) <= 1000:
        out_dict['32_64'].append(car_info[i])
    elif 64 < density[i] <= 128 and len(out_dict['64_128']) <= 1000:
        out_dict['64_128'].append(car_info[i])
    elif 128 < density[i] and len(out_dict['128']) <= 1000:
        out_dict['128'].append(car_info[i])
    else:
        continue



# x = car_info[0]
# print(x)
with open(os.path.join(root, os.path.join(root, out_file)), 'wb') as f1:
    pickle.dump(out_dict, f1)
