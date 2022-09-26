import argparse
import glob
import os.path
from pathlib import Path
import pickle
try:
    import open3d
    from visual_utils import open3d_vis_utils as V
    OPEN3D_FLAG = True
except:
    import mayavi.mlab as mlab
    from visual_utils import visualize_utils as V
    OPEN3D_FLAG = False

import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from pcdet.datasets.waymo.waymo_dataset import WaymoDataset


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        """
        Args:
            root_path:
            dataset_cfg:
            class_names:
            training:
            logger:
        """
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.root_path = root_path
        self.ext = ext
        data_file_list = glob.glob(str(root_path / f'*{self.ext}')) if self.root_path.is_dir() else [self.root_path]

        data_file_list.sort()
        self.sample_file_list = data_file_list

    def __len__(self):
        return len(self.sample_file_list)

    def __getitem__(self, index):
        if self.ext == '.bin':
            points = np.fromfile(self.sample_file_list[index], dtype=np.float32).reshape(-1, 4)
        elif self.ext == '.npy':
            points = np.load(self.sample_file_list[index])
        else:
            raise NotImplementedError

        input_dict = {
            'points': points,
            'frame_id': index,
        }

        data_dict = self.prepare_data(data_dict=input_dict)
        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/waymo_models/centerpoint_pillar_car_base_b2_vis.yaml',
                        help='specify the config for demo')
    parser.add_argument('--data_path', type=str, default='/media/didi/1.0TB/waymo',
                        help='specify the point cloud data file or directory')
    parser.add_argument('--ckpt', type=str, default='/home/didi/Downloads/OpenPCDet-master/output/waymo_models/demo_ckpt/checkpoint_epoch_30.pth', help='specify the pretrained model')
    parser.add_argument('--ext', type=str, default='.bin', help='specify the extension of your point cloud data file')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    # args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')

    # demo_dataset = WaymoDataset(dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
    #     root_path=Path(args.data_path), logger=logger)
    # logger.info(f'Total number of samples: \t{len(demo_dataset)}')

    root = '/media/didi/1.0TB/waymo'
    file = 'dbinfos_car_by_num_points_in_gt_distance.pkl'
    info_path = os.path.join(root, file)
    with open(info_path, 'rb') as f:
        db_infos = pickle.load(f)
    vis_info = db_infos['10_20'][:-1]
    show_num = 1
    distance_amplify = 1.5

    for i in range(0, len(vis_info), show_num):
        npgt_list = []
        gt_box_list = []
        points_list = []
        for j in range(i, i+show_num):
            point_path = os.path.join(root, vis_info[j]['path'])
            num_points_in_gt = vis_info[j]['num_points_in_gt']
            gt_box = vis_info[j]['box3d_lidar']

            obj_points = np.fromfile(str(point_path), dtype=np.float32).reshape(
                [-1, 5])
            if obj_points.shape[0] != num_points_in_gt:
                obj_points = np.fromfile(str(point_path), dtype=np.float64).reshape(
                    [-1, 5])

            obj_points[:, 2] += gt_box[2].astype(np.float32)
            obj_points[:, :2] += gt_box[:2].astype(np.float32) * distance_amplify

            gt_box[:2] = gt_box[:2] * distance_amplify

            gt_box = torch.from_numpy(gt_box).float().unsqueeze(0)
            npgt_list.append(num_points_in_gt)
            points_list.append(obj_points)
            gt_box_list.append(gt_box)

        all_obj_points = np.concatenate(points_list, axis=0)
        all_npgt = np.array(npgt_list)
        all_boxes = torch.concat(gt_box_list, dim=0)

        print('Current_index:', i)
        V.draw_scenes(
            points=all_obj_points, gt_boxes=all_boxes
        )

    logger.info('Demo done.')


if __name__ == '__main__':
    main()
