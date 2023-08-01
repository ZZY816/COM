import numpy as np
import torch

from ....ops.iou3d_nms import iou3d_nms_utils
from ....utils import box_utils


class CurriculumAxisAlignedTargetAssigner(object):
    def __init__(self, model_cfg, class_names, box_coder, match_height=False):
        super().__init__()

        anchor_generator_cfg = model_cfg.ANCHOR_GENERATOR_CONFIG
        anchor_target_cfg = model_cfg.TARGET_ASSIGNER_CONFIG
        self.box_coder = box_coder
        self.match_height = match_height
        self.class_names = np.array(class_names)
        self.anchor_class_names = [config['class_name'] for config in anchor_generator_cfg]
        self.pos_fraction = anchor_target_cfg.POS_FRACTION if anchor_target_cfg.POS_FRACTION >= 0 else None
        self.sample_size = anchor_target_cfg.SAMPLE_SIZE
        self.norm_by_num_examples = anchor_target_cfg.NORM_BY_NUM_EXAMPLES
        self.matched_thresholds = {}
        self.unmatched_thresholds = {}
        for config in anchor_generator_cfg:
            self.matched_thresholds[config['class_name']] = config['matched_threshold']
            self.unmatched_thresholds[config['class_name']] = config['unmatched_threshold']

        self.use_multihead = model_cfg.get('USE_MULTIHEAD', False)
        # self.separate_multihead = model_cfg.get('SEPARATE_MULTIHEAD', False)
        # if self.seperate_multihead:
        #     rpn_head_cfgs = model_cfg.RPN_HEAD_CFGS
        #     self.gt_remapping = {}
        #     for rpn_head_cfg in rpn_head_cfgs:
        #         for idx, name in enumerate(rpn_head_cfg['HEAD_CLS_NAME']):
        #             self.gt_remapping[name] = idx + 1



    def assign_targets(self, all_anchors, gt_boxes_with_classes, group=None):
        """
        Args:
            all_anchors: [(N, 7), ...]
            gt_boxes: (B, M, 8)
        Returns:

        """
        bbox_targets = []
        cls_labels = []
        reg_weights = []
        groups = []

        batch_size = gt_boxes_with_classes.shape[0]
        gt_classes = gt_boxes_with_classes[:, :, -1]
        gt_boxes = gt_boxes_with_classes[:, :, :-1]

        assert gt_classes.shape == group.shape


        for k in range(batch_size):
            cur_gt = gt_boxes[k]
            cnt = cur_gt.__len__() - 1
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            cur_gt = cur_gt[:cnt + 1]

            cur_gt_classes = gt_classes[k][:cnt + 1].int()
            cur_group = group[k][:cnt + 1].int()
            # print(cur_gt_classes.shape)
            # print(cur_group.shape)
            # exit()

            target_list = []
            for anchor_class_name, anchors in zip(self.anchor_class_names, all_anchors):
                if cur_gt_classes.shape[0] > 1:
                    mask = torch.from_numpy(self.class_names[cur_gt_classes.cpu() - 1] == anchor_class_name)
                else:
                    mask = torch.tensor([self.class_names[c - 1] == anchor_class_name
                                         for c in cur_gt_classes], dtype=torch.bool)

                if self.use_multihead:
                    anchors = anchors.permute(3, 4, 0, 1, 2, 5).contiguous().view(-1, anchors.shape[-1])
                    # if self.seperate_multihead:
                    #     selected_classes = cur_gt_classes[mask].clone()
                    #     if len(selected_classes) > 0:
                    #         new_cls_id = self.gt_remapping[anchor_class_name]
                    #         selected_classes[:] = new_cls_id
                    # else:
                    #     selected_classes = cur_gt_classes[mask]
                    selected_classes = cur_gt_classes[mask]
                    selected_group = cur_group[mask]
                else:
                    feature_map_size = anchors.shape[:3]
                    anchors = anchors.view(-1, anchors.shape[-1])
                    selected_classes = cur_gt_classes[mask]
                    selected_group = cur_group[mask]


                single_target = self.assign_targets_single(
                    anchors,
                    cur_gt[mask],
                    gt_classes=selected_classes,
                    group=selected_group,
                    matched_threshold=self.matched_thresholds[anchor_class_name],
                    unmatched_threshold=self.unmatched_thresholds[anchor_class_name]
                )
                target_list.append(single_target)

            if self.use_multihead:
                target_dict = {
                    'box_cls_labels': [t['box_cls_labels'].view(-1) for t in target_list],
                    'box_reg_targets': [t['box_reg_targets'].view(-1, self.box_coder.code_size) for t in target_list],
                    'reg_weights': [t['reg_weights'].view(-1) for t in target_list]
                }

                target_dict['box_reg_targets'] = torch.cat(target_dict['box_reg_targets'], dim=0)
                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=0).view(-1)
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=0).view(-1)
            else:
                target_dict = {
                    'box_cls_labels': [t['box_cls_labels'].view(*feature_map_size, -1) for t in target_list],
                    'box_reg_targets': [t['box_reg_targets'].view(*feature_map_size, -1, self.box_coder.code_size)
                                        for t in target_list],
                    'reg_weights': [t['reg_weights'].view(*feature_map_size, -1) for t in target_list],
                    'groups': [t['groups'].view(*feature_map_size, -1) for t in target_list]
                }
                target_dict['box_reg_targets'] = torch.cat(
                    target_dict['box_reg_targets'], dim=-2
                ).view(-1, self.box_coder.code_size)

                target_dict['box_cls_labels'] = torch.cat(target_dict['box_cls_labels'], dim=-1).view(-1)
                target_dict['reg_weights'] = torch.cat(target_dict['reg_weights'], dim=-1).view(-1)
                target_dict['groups'] = torch.cat(target_dict['groups'], dim=-1).view(-1)

            bbox_targets.append(target_dict['box_reg_targets'])
            cls_labels.append(target_dict['box_cls_labels'])
            reg_weights.append(target_dict['reg_weights'])
            groups.append(target_dict['groups'])


        bbox_targets = torch.stack(bbox_targets, dim=0)

        cls_labels = torch.stack(cls_labels, dim=0)
        reg_weights = torch.stack(reg_weights, dim=0)
        groups = torch.stack(groups, dim=0)
        all_targets_dict = {
            'box_cls_labels': cls_labels,
            'box_reg_targets': bbox_targets,
            'reg_weights': reg_weights,
            'groups': groups

        }

        return all_targets_dict

    def deter_group(self, group, density, distance, angle, density_threshold, angle_threshold):

        distance_condition_list = [(distance <= 30), (distance > 30) & (distance <= 50), (distance > 50)]
        density_condition_list = []
        for i in range(3):
            density_condition_list.append((density > density_threshold[2 * i+1]))
            density_condition_list.append((density_threshold[2 * i] < density) & (density <= density_threshold[2 * i+1]))
            density_condition_list.append((density <= density_threshold[2 * i]))

        angle_condition_list = []
        for i in range(9):
            angle_condition_list.append((angle > angle_threshold[2 * i+1]))
            angle_condition_list.append((angle_threshold[2 * i] < angle) & (angle <= angle_threshold[2 * i+1]))
            angle_condition_list.append((angle <= angle_threshold[2 * i]))

        for i in range(27):
            distance_condition = distance_condition_list[i//9]
            density_condition = density_condition_list[i//3]
            angle_condition = angle_condition_list[i]
            group[distance_condition & density_condition & angle_condition] = i+1

        return group

    def group_classifier(self, gt_boxes, gt_classes, true_object, num_points_in_gt):

        x = gt_boxes[:, 0]
        y = gt_boxes[:, 1]
        dx = gt_boxes[:, 3]
        dy = gt_boxes[:, 4]
        dz = gt_boxes[:, 5]
        yaw = gt_boxes[:, 6]

        distance = torch.sqrt((torch.pow(x, 2) + torch.pow(y, 2)))
        volume = dx * dy * dz
        density = num_points_in_gt / volume
        angle = torch.arccos(
            torch.abs(x * torch.cos(yaw) + y * torch.sin(yaw)) / distance) / torch.pi * 180

        class_id = gt_classes.max()
        assert sum(gt_classes)/len(gt_classes) == class_id
        group = torch.zeros(gt_classes.shape, dtype=torch.int32, device=gt_classes.device)
        if class_id == 1:
            density_threshold = [25.586131140243143, 74.8476162739795, 3.8359461810012303, 9.29043139291457,
                                 1.2967496342110914, 3.095990507938969]
            angle_threshold = [25.51633492714033, 50.8216391042358, 15.209189925724859, 38.31030681069222,
                               9.649144610135792, 31.92730394842723, 15.134228303833503, 47.4479079854979,
                               9.106789882607913, 32.848219367143585, 9.221392855749356, 40.76371342509199,
                               10.770206019384348, 48.66745681357035, 7.417438911885786, 32.710158943251734,
                               7.337952163727097, 38.62494112579102]

            group = self.deter_group(group, density, distance, angle, density_threshold, angle_threshold)

        elif class_id == 2:
            density_threshold = [65.83663696175651, 144.4075229860475, 19.65950740692606, 35.69752426703042,
                                 9.501407481168394, 16.46451534526001]
            angle_threshold = [33.17552208738987, 59.24205440483846, 27.850208190240714, 53.10993988176249,
                               26.11748442240413, 51.98823601184026, 16.295801230541997, 40.69270799878644,
                               16.31452619046975, 42.215708846510694, 17.44845296603078, 44.68977230176953,
                               11.223139395443663, 32.63591000952351, 10.69579687795296, 30.207041264761624,
                               10.978724364950745, 33.90203169147615]

            group = self.deter_group(group, density, distance, angle, density_threshold, angle_threshold)

        elif class_id == 3:
            density_threshold = [57.461388759796236, 134.9975893827867, 15.922123951377536, 25.997081017796724,
                                 6.790408800947857, 11.123947365078406]
            angle_threshold = [32.36623702655827, 61.32386043022792, 20.30528449321854, 47.41579613386567,
                               15.260138668963435, 39.73289221693422, 24.958757841951673, 58.47198672408853,
                               13.451297403447493, 37.21414483587354, 13.556402361155378, 45.26282794663588,
                               10.370940047904533, 53.56355162562699, 5.113336835938596, 21.847996555732493,
                               8.719450202425074, 34.946203011409445]

            group = self.deter_group(group, density, distance, angle, density_threshold, angle_threshold)

        group[true_object == 2] = -1

        return group


    def assign_targets_single(self, anchors, gt_boxes, gt_classes, group, matched_threshold=0.6, unmatched_threshold=0.45):

        assert gt_classes.shape == group.shape


        # if len(gt_boxes) > 0 and anchors.shape[0] > 0:
        #     gp = self.group_classifier(gt_boxes, gt_classes, true_object, num_points_in_gt)
        # else:
        #     gp = None

        num_anchors = anchors.shape[0]
        num_gt = gt_boxes.shape[0]

        labels = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1
        gt_ids = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1
        groups = torch.ones((num_anchors,), dtype=torch.int32, device=anchors.device) * -1

        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            anchor_by_gt_overlap = iou3d_nms_utils.boxes_iou3d_gpu(anchors[:, 0:7], gt_boxes[:, 0:7]) \
                if self.match_height else box_utils.boxes3d_nearest_bev_iou(anchors[:, 0:7], gt_boxes[:, 0:7])

            # NOTE: The speed of these two versions depends the environment and the number of anchors
            # anchor_to_gt_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=1)).cuda()
            anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(dim=1)
            anchor_to_gt_max = anchor_by_gt_overlap[torch.arange(num_anchors, device=anchors.device), anchor_to_gt_argmax]

            # gt_to_anchor_argmax = torch.from_numpy(anchor_by_gt_overlap.cpu().numpy().argmax(axis=0)).cuda()
            gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(dim=0)
            gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax, torch.arange(num_gt, device=anchors.device)]
            empty_gt_mask = gt_to_anchor_max == 0
            gt_to_anchor_max[empty_gt_mask] = -1

            anchors_with_max_overlap = (anchor_by_gt_overlap == gt_to_anchor_max).nonzero()[:, 0]
            gt_inds_force = anchor_to_gt_argmax[anchors_with_max_overlap]
            labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
            gt_ids[anchors_with_max_overlap] = gt_inds_force.int()
            groups[anchors_with_max_overlap] = group[gt_inds_force]

            pos_inds = anchor_to_gt_max >= matched_threshold
            gt_inds_over_thresh = anchor_to_gt_argmax[pos_inds]
            labels[pos_inds] = gt_classes[gt_inds_over_thresh]
            gt_ids[pos_inds] = gt_inds_over_thresh.int()
            bg_inds = (anchor_to_gt_max < unmatched_threshold).nonzero()[:, 0]
            groups[pos_inds] = group[gt_inds_over_thresh]

            # print(labels.shape)
            # print(labels)
            # print(groups)
            # print((labels >= 0).sum())
            # print((groups >= 0).sum())
            # exit()

        else:
            bg_inds = torch.arange(num_anchors, device=anchors.device)

        fg_inds = (labels > 0).nonzero()[:, 0]

        if self.pos_fraction is not None:
            num_fg = int(self.pos_fraction * self.sample_size)
            if len(fg_inds) > num_fg:
                num_disabled = len(fg_inds) - num_fg
                disable_inds = torch.randperm(len(fg_inds))[:num_disabled]
                labels[disable_inds] = -1
                fg_inds = (labels > 0).nonzero()[:, 0]

            num_bg = self.sample_size - (labels > 0).sum()
            if len(bg_inds) > num_bg:
                enable_inds = bg_inds[torch.randint(0, len(bg_inds), size=(num_bg,))]
                labels[enable_inds] = 0
            # bg_inds = torch.nonzero(labels == 0)[:, 0]
        else:
            if len(gt_boxes) == 0 or anchors.shape[0] == 0:
                labels[:] = 0
                groups[:] = 0
            else:
                labels[bg_inds] = 0
                labels[anchors_with_max_overlap] = gt_classes[gt_inds_force]
                groups[bg_inds] = 0
                groups[anchors_with_max_overlap] = group[gt_inds_force]

        bbox_targets = anchors.new_zeros((num_anchors, self.box_coder.code_size))
        if len(gt_boxes) > 0 and anchors.shape[0] > 0:
            fg_gt_boxes = gt_boxes[anchor_to_gt_argmax[fg_inds], :]
            fg_anchors = anchors[fg_inds, :]
            bbox_targets[fg_inds, :] = self.box_coder.encode_torch(fg_gt_boxes, fg_anchors)

        reg_weights = anchors.new_zeros((num_anchors,))

        if self.norm_by_num_examples:
            num_examples = (labels >= 0).sum()
            num_examples = num_examples if num_examples > 1.0 else 1.0
            reg_weights[labels > 0] = 1.0 / num_examples
        else:
            reg_weights[labels > 0] = 1.0

        # print(labels.shape)
        # print(labels)
        # print(groups)
        # print((labels > 0).sum())
        # print((groups > 0).sum())
        # exit()


        ret_dict = {
            'box_cls_labels': labels,
            'box_reg_targets': bbox_targets,
            'reg_weights': reg_weights,
            'groups': groups
        }
        return ret_dict
