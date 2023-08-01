import copy
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from ..model_utils import model_nms_utils
from ..model_utils import centernet_utils
from ...utils import loss_utils
from .curriculum_center_head import SeparateHead, CurriculumCenterHead
from .curri_anchor_head_single import CurriculumAnchorHeadSingle

class CurriculumAnchorHeadSingle_x1(CurriculumAnchorHeadSingle):
    def cluster(self, gt_boxes, true_object, occupancy_ratio, facade_type):
        distance = torch.sqrt(torch.pow(gt_boxes[:, :, 0], 2) + torch.pow(gt_boxes[:, :, 1], 2))
        length = gt_boxes[:, :, 3]
        class_id = gt_boxes[:, :, -1]
        occupancy_ratio = occupancy_ratio
        facade_type = facade_type

        group = gt_boxes.new_zeros(class_id.shape).long()

        # print(len(class_id[(class_id==1) & (true_object==2)]))

        distance_condition_list = [(distance <= 15), (distance > 15) & (distance <= 30), (distance > 30) & (distance <= 45), (distance > 45) & (distance <= 60),
                                   (distance > 60)]
        length_condition_list = [(length <= 6), (length > 6)]
        facade_condition_list = [(facade_type == 3), (facade_type == 2), (facade_type == 1), (facade_type == 0)]
        occupancy_condition_list = [(occupancy_ratio <= 0.21 * 5 / 12),
                                    (occupancy_ratio <= 0.41 * 5 / 12) & (occupancy_ratio > 0.21 * 5 / 12),
                                    (occupancy_ratio <= 0.61 * 5 / 12) & (occupancy_ratio > 0.41 * 5 / 12)
                                       , (occupancy_ratio <= 0.81 * 5 / 12) & (occupancy_ratio > 0.61 * 5 / 12),
                                    (occupancy_ratio > 0.81 * 5 / 12)][::-1]
        occupancy_car_condition_list = [(occupancy_ratio <= 0.25),
                                        (occupancy_ratio <= 0.5) & (occupancy_ratio > 0.25)
                                           , (occupancy_ratio <= 0.7) & (occupancy_ratio > 0.5),
                                        (occupancy_ratio > 0.7)][::-1]

        car_group = 1
        for distance_condition in distance_condition_list:
            for length_condition in length_condition_list:
                for facade_condition in facade_condition_list:
                    for occupancy_condition in occupancy_car_condition_list:
                        group[distance_condition & length_condition & facade_condition & occupancy_condition & (
                                class_id == 1) & (true_object == 1)] = car_group
                        car_group += 1

        ped_group = 1
        for distance_condition in distance_condition_list:
            for occupancy_condition in occupancy_condition_list:
                if class_id.max() == 1:
                    group[distance_condition & occupancy_condition & (class_id == 1) & (true_object == 1)] = ped_group
                else:
                    group[distance_condition & occupancy_condition & (class_id == 2) & (true_object == 1)] = ped_group
                ped_group += 1

        cyc_group = 1
        for distance_condition in distance_condition_list:
            for occupancy_condition in occupancy_condition_list:
                if class_id.max() == 1:
                    group[distance_condition & occupancy_condition & (class_id == 1) & (true_object == 1)] = cyc_group
                else:
                    group[distance_condition & occupancy_condition & (class_id == 3) & (true_object == 1)] = cyc_group
                cyc_group += 1

        return group


class CurriculumAnchorHeadSingle_car(CurriculumAnchorHeadSingle):
    def cluster(self, gt_boxes, true_object, occupancy_ratio, facade_type):

        distance = torch.sqrt(torch.pow(gt_boxes[:, :, 0], 2) + torch.pow(gt_boxes[:, :, 1], 2))
        length = gt_boxes[:, :, 3]
        class_id = gt_boxes[:, :, -1]
        occupancy_ratio = occupancy_ratio
        facade_type = facade_type

        group = gt_boxes.new_zeros(class_id.shape).long()

        # print(len(class_id[(class_id==1) & (true_object==2)]))

        distance_condition_list = [(distance <= 30), (distance > 30) & (distance <= 50),
                                   (distance > 50)]
        length_condition_list = [(length <= 6), (length > 6)]
        facade_condition_list = [(facade_type == 3), (facade_type == 2), (facade_type == 1), (facade_type == 0)]
        occupancy_condition_list = [(occupancy_ratio <= 0.21 * 5 / 12),
                                    (occupancy_ratio <= 0.41 * 5 / 12) & (occupancy_ratio > 0.21 * 5 / 12),
                                    (occupancy_ratio <= 0.61 * 5 / 12) & (occupancy_ratio > 0.41 * 5 / 12)
                                       , (occupancy_ratio <= 0.81 * 5 / 12) & (occupancy_ratio > 0.61 * 5 / 12),
                                    (occupancy_ratio > 0.81 * 5 / 12)][::-1]
        occupancy_car_condition_list = [(occupancy_ratio <= 0.25),
                                        (occupancy_ratio <= 0.5) & (occupancy_ratio > 0.25)
                                           , (occupancy_ratio <= 0.7) & (occupancy_ratio > 0.5),
                                        (occupancy_ratio > 0.7)][::-1]

        car_group = 1
        for distance_condition in distance_condition_list:
            for length_condition in length_condition_list:
                for facade_condition in facade_condition_list:
                    for occupancy_condition in occupancy_car_condition_list:
                        group[distance_condition & length_condition & facade_condition & occupancy_condition & (
                                class_id == 1) & (true_object == 1)] = car_group
                        car_group += 1

        return group


class CurriculumAnchorHeadSingle_car_x2(CurriculumAnchorHeadSingle):
    def cluster(self, gt_boxes, true_object, occupancy_ratio, facade_type):

        distance = torch.sqrt(torch.pow(gt_boxes[:, :, 0], 2) + torch.pow(gt_boxes[:, :, 1], 2))
        length = gt_boxes[:, :, 3]
        class_id = gt_boxes[:, :, -1]
        occupancy_ratio = occupancy_ratio
        facade_type = facade_type

        group = gt_boxes.new_zeros(class_id.shape).long()

        # print(len(class_id[(class_id==1) & (true_object==2)]))

        distance_condition_list = [(distance <= 30), (distance > 30) & (distance <= 50),
                                   (distance > 50)]
        length_condition_list = [(length <= 6), (length > 6)]
        facade_condition_list = [(facade_type == 3), (facade_type == 2), (facade_type == 1), (facade_type == 0)]
        occupancy_condition_list = [(occupancy_ratio <= 0.21),
                                    (occupancy_ratio <= 0.41) & (occupancy_ratio > 0.21),
                                    (occupancy_ratio <= 0.61) & (occupancy_ratio > 0.41)
                                       , (occupancy_ratio <= 0.81) & (occupancy_ratio > 0.61),
                                    (occupancy_ratio > 0.81)][::-1]
        # occupancy_car_condition_list = [(occupancy_ratio <= 0.25),
        #                                 (occupancy_ratio <= 0.5) & (occupancy_ratio > 0.25)
        #                                    , (occupancy_ratio <= 0.7) & (occupancy_ratio > 0.5),
        #                                 (occupancy_ratio > 0.7)][::-1]

        car_group = 1
        for distance_condition in distance_condition_list:
            for occupancy_condition in occupancy_condition_list:
                group[distance_condition & occupancy_condition & (class_id == 1) & (true_object == 1)] = car_group
                car_group += 1

        return group




class CurriculumCenterHead_x5(CurriculumCenterHead):
    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossCenterCurriculum(self.model_cfg, conf_shape=(3, 96)))
        self.add_module('reg_loss_func', loss_utils.RegLossCenterNet())



class CurriculumCenterHead_ped_merge(CurriculumCenterHead):

    def cluster(self, gt_boxes, true_object, occupancy_ratio, facade_type):
        distance = torch.sqrt(torch.pow(gt_boxes[:, :, 0], 2) + torch.pow(gt_boxes[:, :, 1], 2))
        length = gt_boxes[:, :, 3]
        class_id = gt_boxes[:, :, -1]
        occupancy_ratio = occupancy_ratio
        facade_type = facade_type

        group = gt_boxes.new_zeros(class_id.shape).long()

        # print(len(class_id[(class_id==1) & (true_object==2)]))


        distance_condition_list = [(distance <= 30), (distance > 30) & (distance <= 50),
                                   (distance > 50)]
        length_condition_list = [(length <= 6), (length > 6)]
        facade_condition_list = [(facade_type == 3), (facade_type == 2), (facade_type == 1), (facade_type == 0)]
        occupancy_condition_list = [(occupancy_ratio <= 0.21 * 5/12), (occupancy_ratio <= 0.41 * 5/12) & (occupancy_ratio > 0.21 * 5/12), (occupancy_ratio <= 0.61 * 5/12) & (occupancy_ratio > 0.41 * 5/12)
            , (occupancy_ratio <= 0.81 * 5/12) & (occupancy_ratio > 0.61 * 5/12), (occupancy_ratio > 0.81 * 5/12)][::-1]

        ped_group = 1
        for distance_condition in distance_condition_list:
            for occupancy_condition in occupancy_condition_list:
                group[distance_condition & occupancy_condition & (class_id == 1) & (true_object==1)] = ped_group
                ped_group += 1
        # print(true_object)
        # print(group)
        # exit()
        return group

    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossCenterCurriculumMerge(self.model_cfg, conf_shape=(1, 15)))
        self.add_module('reg_loss_func', loss_utils.RegLossCenterNet())


class CurriculumCenterHead_car_merge(CurriculumCenterHead):
    def build_losses(self):
        self.add_module('hm_loss_func', loss_utils.FocalLossCenterCurriculumMerge(self.model_cfg, conf_shape=(1, 96)))
        self.add_module('reg_loss_func', loss_utils.RegLossCenterNet())


