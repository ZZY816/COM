import numpy as np
import torch.nn as nn

from .anchor_head_curriculum import AnchorHeadCurriculum
import torch

class CurriculumAnchorHeadSingle(AnchorHeadCurriculum):
    def __init__(self, model_cfg, input_channels, num_class, class_names, grid_size, point_cloud_range,
                 predict_boxes_when_training=True, **kwargs):
        super().__init__(
            model_cfg=model_cfg, num_class=num_class, class_names=class_names, grid_size=grid_size, point_cloud_range=point_cloud_range,
            predict_boxes_when_training=predict_boxes_when_training
        )

        self.num_anchors_per_location = sum(self.num_anchors_per_location)

        self.conv_cls = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.num_class,
            kernel_size=1
        )
        self.conv_box = nn.Conv2d(
            input_channels, self.num_anchors_per_location * self.box_coder.code_size,
            kernel_size=1
        )

        if self.model_cfg.get('USE_DIRECTION_CLASSIFIER', None) is not None:
            self.conv_dir_cls = nn.Conv2d(
                input_channels,
                self.num_anchors_per_location * self.model_cfg.NUM_DIR_BINS,
                kernel_size=1
            )
        else:
            self.conv_dir_cls = None
        self.init_weights()

        self.epoch = 0

    def init_weights(self):
        pi = 0.01
        nn.init.constant_(self.conv_cls.bias, -np.log((1 - pi) / pi))
        nn.init.normal_(self.conv_box.weight, mean=0, std=0.001)

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

        # car_group = 1
        # for distance_condition in distance_condition_list:
        #     for length_condition in length_condition_list:
        #         for facade_condition in facade_condition_list:
        #             for occupancy_condition in occupancy_car_condition_list:
        #                 group[distance_condition & length_condition & facade_condition & occupancy_condition & (
        #                             class_id == 1) & (true_object == 1)] = car_group
        #                 car_group += 1

        ped_group = 1
        for distance_condition in distance_condition_list:
            for occupancy_condition in occupancy_condition_list:
                if class_id.max() == 1:
                    group[distance_condition & occupancy_condition & (class_id == 1) & (true_object == 1)] = ped_group
                else:
                    group[distance_condition & occupancy_condition & (class_id == 2) & (true_object == 1)] = ped_group
                ped_group += 1

        # cyc_group = 1
        # for distance_condition in distance_condition_list:
        #     for occupancy_condition in occupancy_condition_list:
        #         if class_id.max() == 1:
        #             group[distance_condition & occupancy_condition & (class_id == 1) & (true_object == 1)] = cyc_group
        #         else:
        #             group[distance_condition & occupancy_condition & (class_id == 3) & (true_object == 1)] = cyc_group
        #         cyc_group += 1

        return group

    def forward(self, data_dict):

        spatial_features_2d = data_dict['spatial_features_2d']

        cls_preds = self.conv_cls(spatial_features_2d)
        box_preds = self.conv_box(spatial_features_2d)

        cls_preds = cls_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C] torch.Size([2, 468, 468, 18])
        box_preds = box_preds.permute(0, 2, 3, 1).contiguous()  # [N, H, W, C] torch.Size([2, 468, 468, 42])

        self.forward_ret_dict['cls_preds'] = cls_preds
        self.forward_ret_dict['box_preds'] = box_preds


        if self.conv_dir_cls is not None:
            dir_cls_preds = self.conv_dir_cls(spatial_features_2d)
            dir_cls_preds = dir_cls_preds.permute(0, 2, 3, 1).contiguous()
            self.forward_ret_dict['dir_cls_preds'] = dir_cls_preds
        else:
            dir_cls_preds = None



        #print(cls_preds.shape, box_preds.shape, data_dict.keys())
        if self.training:
            group = self.cluster(gt_boxes=data_dict['gt_boxes'], true_object=data_dict['true_object'],
                                 occupancy_ratio=data_dict['occupancy_ratio'], facade_type=data_dict['facade_type'])
            targets_dict = self.assign_targets(
                gt_boxes=data_dict['gt_boxes'], group=group,
            )
            self.forward_ret_dict.update(targets_dict)

        if not self.training or self.predict_boxes_when_training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=data_dict['batch_size'],
                cls_preds=cls_preds, box_preds=box_preds, dir_cls_preds=dir_cls_preds
            )
            data_dict['batch_cls_preds'] = batch_cls_preds
            data_dict['batch_box_preds'] = batch_box_preds
            data_dict['cls_preds_normalized'] = False

        return data_dict
