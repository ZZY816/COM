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
print(len(npgt[(npgt > 0) & (npgt < 5)]))

out_file = 'dbinfos_car_by_num_points_in_gt_random.pkl'
out_dict = {}
out_dict['10000'] = []
out_dict['5000_10000'] = []
out_dict['2000_5000'] = []
out_dict['1000_2000'] = []
out_dict['500_1000'] = []
out_dict['250_500'] = []
out_dict['150_250'] = []
out_dict['100_150'] = []
out_dict['50_100'] = []
out_dict['25_50'] = []
out_dict['10_25'] = []
out_dict['5_10'] = []
out_dict['0_5'] = []

for i in np.random.permutation(len(car_info)):
    if i%1000 ==0:
        print(i)
    if npgt[i] == 0:
        continue
    if npgt[i] > 10000 and len(out_dict['10000']) <= 1000:
        out_dict['10000'].append(car_info[i])
    elif 5000 < npgt[i] <= 10000 and len(out_dict['5000_10000']) <= 1000:
        out_dict['5000_10000'].append(car_info[i])
    elif 2000 < npgt[i] <= 5000 and len(out_dict['2000_5000']) <= 1000:
        out_dict['2000_5000'].append(car_info[i])
    elif 1000 < npgt[i] <= 2000 and len(out_dict['1000_2000']) <= 1000:
        out_dict['1000_2000'].append(car_info[i])
    elif 500 < npgt[i] <= 1000 and len(out_dict['500_1000']) <= 1000:
        out_dict['500_1000'].append(car_info[i])
    elif 250 < npgt[i] <= 500 and len(out_dict['250_500']) <= 1000:
        out_dict['250_500'].append(car_info[i])
    elif 150 < npgt[i] <= 250 and len(out_dict['150_250']) <= 1000:
        out_dict['150_250'].append(car_info[i])
    elif 100 < npgt[i] <= 150 and len(out_dict['100_150']) <= 1000:
        out_dict['100_150'].append(car_info[i])
    elif 50 < npgt[i] <= 100 and len(out_dict['50_100']) <= 1000:
        out_dict['50_100'].append(car_info[i])
    elif 25 < npgt[i] <= 50 and len(out_dict['25_50']) <= 1000:
        out_dict['25_50'].append(car_info[i])
    elif 10 < npgt[i] <= 25 and len(out_dict['10_25']) <= 1000:
        out_dict['10_25'].append(car_info[i])
    elif 5 < npgt[i] <= 10 and len(out_dict['5_10']) <= 1000:
        out_dict['5_10'].append(car_info[i])
    elif 0 < npgt[i] <= 5 and len(out_dict['0_5']) <= 1000:
        out_dict['0_5'].append(car_info[i])
    else:
        continue



# x = car_info[0]
# print(x)
with open(os.path.join(root, os.path.join(root, out_file)), 'wb') as f1:
    pickle.dump(out_dict, f1)
