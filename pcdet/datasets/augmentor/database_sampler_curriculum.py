import pickle

import os
import copy
import numpy as np
from skimage import io
import torch
import SharedArray
import torch.distributed as dist

from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils import box_utils, common_utils, calibration_kitti
from pcdet.datasets.kitti.kitti_object_eval_python import kitti_common
from .database_sampler_v2 import DataBaseSampler
import random

class DataBaseSampler_COM1(DataBaseSampler):
    def __init__(self, root_path, sampler_cfg, class_names, logger=None):

        super(DataBaseSampler_COM1, self).__init__(root_path=root_path, sampler_cfg=sampler_cfg, class_names=class_names, logger=logger)
        self.confidence_groups = None
        self.stop = self.sampler_cfg.get('STOP', None)
        self.sigma = self.sampler_cfg.get('SIGMA', 0.1)
        self.mu = self.sampler_cfg.get('MU', 1.5)
        self.mu_pc = self.sampler_cfg.get('MU_PC', 0.3)
        self.ave_epoch = self.sampler_cfg.get('AVE', 100)
        self.sigmapc = self.sampler_cfg.get('SIGMAPC', 0.1)
        self.s3 = self.sampler_cfg.get('S3', [0.1, 0.1, 0.1])
        self.m3 = self.sampler_cfg.get('M3', [1.5, 0.3, 0.3])
        self.anti = self.sampler_cfg.get('ANTI', False)
        self.back = self.sampler_cfg.get('BACK', False)


    def split_groups(self, db_infos, class_name, sample_num):  # important

        def set_threshold(attribute_list, set_num=3):
            threshold_list = []
            for attribute in attribute_list:
                threshold_sub_list = []
                for i in np.arange(1, set_num):
                    threshold_sub_list.append(attribute[np.argsort(attribute)][int(len(attribute) * i / set_num)])
                threshold_list.append(threshold_sub_list)
            return threshold_list

        class_dict = db_infos[class_name]

        num_points_in_gt = np.array([sample['num_points_in_gt'] for sample in class_dict])
        distance = np.array([np.sqrt(np.power(sample['box3d_lidar'][0], 2) +
                                             np.power(sample['box3d_lidar'][1], 2)) for sample in class_dict])
        volume = np.array([sample['box3d_lidar'][3] * sample['box3d_lidar'][4] * sample['box3d_lidar'][5]
                                   for sample in class_dict])
        density = num_points_in_gt / volume
        if class_name == 'Pedestrian':
            occupancy_ratio = np.array([sample['occupancy_ratio'] * 12/5 for sample in class_dict])
        elif class_name == 'Cyclist':
            occupancy_ratio = np.array([sample['occupancy_ratio'] * 12/5 for sample in class_dict])
        else:
            occupancy_ratio = np.array([sample['occupancy_ratio'] for sample in class_dict])
        facade_type = np.array([sample['facade_type'] for sample in class_dict])
        length = np.array([sample['box3d_lidar'][3] for sample in class_dict])

        distance_condition_list = [(distance <= 30), (distance > 30) & (distance <= 50), (distance > 50) & (distance <= 75)]
        length_condition_list = [(length <= 6), (length > 6)]
        facade_condition_list = [(facade_type==3), (facade_type==2), (facade_type==1), (facade_type==0)]
        occupancy_condition_list = [(occupancy_ratio <= 0.21), (occupancy_ratio <= 0.41) & (occupancy_ratio > 0.21), (occupancy_ratio <= 0.61) & (occupancy_ratio > 0.41)
                                    , (occupancy_ratio <= 0.81) & (occupancy_ratio > 0.61), (occupancy_ratio > 0.81)][::-1]
        occupancy_car_condition_list = [(occupancy_ratio <= 0.25),
                                    (occupancy_ratio <= 0.5) & (occupancy_ratio > 0.25)
            , (occupancy_ratio <= 0.7) & (occupancy_ratio > 0.5), (occupancy_ratio > 0.7)][::-1]
        # x = set_threshold([num_points_in_gt[distance_condition_list[0]], num_points_in_gt[distance_condition_list[1]],
        #                    num_points_in_gt[distance_condition_list[2]]], set_num=8)
        condition_list = []
        if class_name == 'Vehicle':
            for distance_condition in distance_condition_list:
                for length_condition in length_condition_list:
                    for facade_conditio in facade_condition_list:
                        for occupancy_condition in occupancy_car_condition_list:
                            condition_list.append(distance_condition & length_condition & facade_conditio & occupancy_condition)
        elif class_name == 'Pedestrian':
            for distance_condition in distance_condition_list:
                for occupancy_condition in occupancy_condition_list:
                    condition_list.append(distance_condition & occupancy_condition)
        elif class_name == 'Cyclist':
            for distance_condition in distance_condition_list:
                for occupancy_condition in occupancy_condition_list:
                    condition_list.append(distance_condition & occupancy_condition)

        group_list = [np.where(condition)[0] for condition in condition_list]

        indices_list = group_list
        pointer_list = [len(group) for group in indices_list]

        if class_name == 'Vehicle':
            x = np.array(pointer_list).reshape(3, 2, 4, 4)
        else:
            x = np.array(pointer_list).reshape(3, 5)

        print(class_name, x)

        sample_group = {
            'sample_num': sample_num,
            'pointer': pointer_list,
            'indices': indices_list
        }

        return sample_group

    def sample_with_fixed_number_v2(self, class_name, sample_group, sample_num_list=None):
        """
        Args:
            class_name:
            sample_group:
        Returns:

        """
        total_sample_num, pointer_list, indices_list = int(sample_group['sample_num']), sample_group['pointer'], \
                                                       sample_group['indices']
        norm = [len(indice) for indice in indices_list]


        group_num = len(pointer_list)
        #probabililty = np.ones(group_num) / group_num
        probabililty = np.array(norm) / sum(norm)

        selected_groups = np.arange(group_num)

        assert selected_groups.shape == probabililty.shape
        real_selected_groups = np.random.choice(selected_groups, total_sample_num, p=probabililty.ravel(), replace=True)

        sampled_dict = []
        for i in real_selected_groups:
            sample_num = 1
            pointer = pointer_list[i]
            indices = indices_list[i]

            if pointer >= len(indices):
                indices = np.random.permutation(indices)
                pointer = 0

            sampled_dict.extend([self.db_infos[class_name][idx] for idx in indices[pointer: pointer + sample_num]])
            pointer += sample_num

            sample_group['pointer'][i] = pointer
            sample_group['indices'][i] = indices

        return sampled_dict




class DataBaseSampler_COM2(DataBaseSampler_COM1):
    def sample_with_fixed_number_v2(self, class_name, sample_group, sample_num_list=None):
        """
        Args:
            class_name: Vehicle. Pedestrian, Cyclist
            sample_group:
        Returns:

        """

        total_sample_num, pointer_list, indices_list = int(sample_group['sample_num']), sample_group['pointer'], \
                                                       sample_group['indices']

        group_num = len(pointer_list)
        selected_groups = np.arange(group_num)
        groups_length = [len(indice) for indice in indices_list]
        total_object_num = sum(groups_length)
        norm = np.array(groups_length)/total_object_num

        if self.confidence_groups is None or self.epoch > self.ave_epoch:
            probabililty = np.ones(group_num) * norm
            probabililty = probabililty / probabililty.sum()

        else:
            # print(self.confidence_groups)
            # exit()
            class_num = self.confidence_groups.shape[0]
            # print(class_num)
            # print(self.confidence_groups[0][:group_num])
            # print(self.confidence_groups[1][:group_num])
            # exit()
            if class_name == 'Vehicle':
                confidence_groups = self.confidence_groups[0][:group_num]

                if self.back:
                    if self.epoch <= 26:
                        k = min(int(self.epoch * self.m3[0]), group_num - 1)
                    else:
                        k = min(int((self.epoch - 26) * self.m3[0]), group_num - 1)
                else:
                    k = min(int(self.epoch * self.m3[0]), group_num-1)
                if self.anti:
                    u = sorted(confidence_groups)[k]  #
                else:
                    u = sorted(confidence_groups)[::-1][k]  #
                # u = confidence_groups.max() + (confidence_groups.min() - confidence_groups.max()) * self.epoch * 0.01
                sigma = np.sqrt(self.s3[0])
                #sigma = np.sqrt(self.s3[0]) if self.epoch <= 26 else np.sqrt(self.s3[0] * 0.1) #

            elif class_name == 'Pedestrian':

                if class_num == 3:
                    confidence_groups = self.confidence_groups[1][:group_num]
                else:
                    confidence_groups = self.confidence_groups[0][:group_num]

                if self.back:
                    if self.epoch <= 26:
                        k = min(int(self.epoch * self.m3[1]), group_num - 1)
                    else:
                        k = min(int((self.epoch - 26) * self.m3[1]), group_num - 1)
                else:
                    k = min(int(self.epoch * self.m3[1]), group_num-1)

                # k = min(int(self.epoch * self.m3[1]), group_num-1)
                if self.anti:
                    u = sorted(confidence_groups)[k]  #
                else:
                    u = sorted(confidence_groups)[::-1][k]  #
                # u = confidence_groups.max() + (confidence_groups.min() - confidence_groups.max()) * self.epoch * 0.01
                sigma = np.sqrt(self.s3[1])
                # sigma = np.sqrt(self.s3[1]) if self.epoch <= 26 else np.sqrt(self.s3[1] * 0.1) #
            else:
                if class_num == 3:
                    confidence_groups = self.confidence_groups[2][:group_num]
                elif class_num == 2:
                    confidence_groups = self.confidence_groups[1][:group_num]
                else:
                    confidence_groups = self.confidence_groups[0][:group_num]

                if self.back:
                    if self.epoch <= 26:
                        k = min(int(self.epoch * self.m3[2]), group_num - 1)
                    else:
                        k = min(int((self.epoch - 26) * self.m3[2]), group_num - 1)
                else:
                    k = min(int(self.epoch * self.m3[2]), group_num-1)

                # k = min(int(self.epoch * self.m3[2]), group_num-1)
                if self.anti:
                    u = sorted(confidence_groups)[k]  #
                else:
                    u = sorted(confidence_groups)[::-1][k]  #
                # u = confidence_groups.max() + (confidence_groups.min() - confidence_groups.max()) * self.epoch * 0.01
                sigma = np.sqrt(self.s3[2])  #
                # sigma = np.sqrt(self.s3[2]) if self.epoch <= 26 else np.sqrt(self.s3[2] * 0.1)

            sample_confidence = np.exp(-(confidence_groups - u) ** 2 / (2 * sigma ** 2)) / (np.sqrt(2 * np.pi) * sigma) * norm
            probabililty = sample_confidence/sample_confidence.sum()



        assert selected_groups.shape == probabililty.shape

        real_selected_groups = np.random.choice(selected_groups, total_sample_num, p=probabililty.ravel(), replace=True)

        sampled_dict = []

        if self.stop is not None:
            if self.epoch >= self.stop:  # stop sampling
                return sampled_dict

        for i in real_selected_groups:
            sample_num = 1
            pointer = pointer_list[i]
            indices = indices_list[i]

            if pointer >= len(indices):
                indices = np.random.permutation(indices)
                pointer = 0

            sampled_dict.extend([self.db_infos[class_name][idx] for idx in indices[pointer: pointer + sample_num]])
            pointer += sample_num

            sample_group['pointer'][i] = pointer
            sample_group['indices'][i] = indices

        return sampled_dict