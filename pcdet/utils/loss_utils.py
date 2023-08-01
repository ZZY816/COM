import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import box_utils
from ..models.model_utils import centernet_utils


class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """

        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights




class CurriculumSigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25, model_config=None):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(CurriculumSigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.confidence_all = 0

        self.model_config = model_config

        self.curriculum = self.model_config.get('LOSS_CURRICULUM', None)
        self.use_curriculum_loss = self.curriculum.get('UCL', True)
        self.oto = self.curriculum.get('OTO', False)
        self.start_epoch = self.curriculum.get('START', 0)
        self.end_epoch = self.curriculum.get('END', 30)
        self.cut_epoch = self.curriculum.get('CUT', 10000)
        self.al = self.curriculum.get('ALPHA', 0.001)
        self.elongation = self.curriculum.get('ELONGATION', -10)
        self.height = self.curriculum.get('HEIGHT', 1)
        self.offset = self.curriculum.get('OFFSET', 0)
        self.inverse = self.curriculum.get('INV', False)
        self.use_norm = self.curriculum.get('NORM', False)
        self.pos_weight = self.curriculum.get('POSW', 1)
        self.fixed = self.curriculum.get('FIXED', False)
        self.merge_scores = self.curriculum.get('MERGE_SCORES', False)
        print(self.start_epoch, self.end_epoch, self.alpha, self.elongation,
              self.height, self.offset, self.inverse, self.merge_scores)
        self.class_num = 3
        self.distribute = self.curriculum.get('DIST', False)

        from scipy.stats import norm
        self.pos_norm = 0.5 / (1 - norm.cdf(self.offset)) * self.pos_weight
        self.neg_norm = 0.5 / norm.cdf(self.offset)

        self.means, self.stds = None, None
        self.sm = self.curriculum.get('SM', False)
        self.sma = self.curriculum.get('SMA', False)
        self.sme = self.curriculum.get('SME', 20)
        self.smt = self.curriculum.get('SMT', 0.15)




    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def update_score(self, pred, gt):

        self.class_num = gt.shape[-1]
        if self.means is None:
            self.means = [None] * self.class_num
            self.stds = [None] * self.class_num

        for idx in range(self.class_num):
            #pos_inds = gt[:, idx, :, :].eq(1)
            #num_obj = pos_inds.float().sum().cpu().detach().numpy()



            if self.distribute:
                scores = pred[:, :, idx][gt[:, :, idx] > 0].cpu().detach().numpy()
                res = [np.sum(scores), np.sum(scores ** 2), len(scores)]
                res = torch.from_numpy(np.array(res)).cuda()
                world_size = torch.distributed.get_world_size()
                res_list = [torch.zeros_like(res) for _ in range(world_size)]
                torch.distributed.all_gather(res_list, res)
                res_list = [e for e in res_list]
                res_list = sum(res_list)
                assert len(res_list) == 3
                src_sum, square_sum, nums = res_list[0], res_list[1], res_list[2]

            else:
                scores = pred[:, :, idx][gt[:, :, idx] > 0].detach()
                res = [torch.sum(scores), torch.sum(scores ** 2), len(scores)]
                src_sum, square_sum, nums = res[0], res[1], res[2]

            if nums == 0:
                return

            mean = src_sum / nums
            if (square_sum + nums * mean ** 2 - 2 * mean * src_sum) <= 0:
                std = 0
            else:
                std = torch.sqrt((square_sum + nums * mean ** 2 - 2 * mean * src_sum) / nums)

            # print( src_sum, square_sum, nums)
            # if square_sum + nums*mean**2 - 2 * mean * src_sum < 0:
            #     exit(1)

            if self.means[idx] is None:
                self.means[idx], self.stds[idx] = mean, std
            else:
                self.means[idx] = (1 - self.alpha) * self.means[idx] + self.alpha * mean
                self.stds[idx] = (1 - self.alpha) * self.stds[idx] + self.alpha * std
        return 0

    def groups_confidence(self, pred, groups):
        num = 96
        class_num = pred.shape[-1]

        confidence = torch.zeros(*list(pred.shape), num + 1, dtype=pred.dtype, device=pred.device)
        groups_num = torch.zeros(*list(pred.shape), num + 1, device=pred.device)
        confidence.scatter_(-1, groups.unsqueeze(dim=-1), pred.unsqueeze(dim=-1))
        groups_num.scatter_(-1, groups.unsqueeze(dim=-1), 1.0)
        # print(confidence[..., 1:][confidence[..., 1:] > 0].shape)
        # print(groups_num[..., 1:][groups_num[..., 1:] > 0].shape)
        # exit()
        confidence = confidence[..., 1:].view(-1, class_num, num).sum(0).detach()
        groups_num = groups_num[..., 1:].view(-1, class_num, num).sum(0)

        return [confidence, groups_num]


    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, groups: torch.Tensor, epoch):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """

        pred_sigmoid = torch.sigmoid(input)

        if self.use_curriculum_loss:
            self.update_score(pred_sigmoid, gt=groups)

            curriculum_weight = torch.ones(pred_sigmoid.shape, device=pred_sigmoid.device)

            for cur_class_id in range(groups.shape[-1]):

                threshold = self.means[cur_class_id] + self.offset * self.stds[cur_class_id]
                var = self.stds[cur_class_id]

                if threshold is None:
                    print(f'threshold is None at epoch {epoch}')
                    threshold = 0.5
                    var = 0.2
                if self.use_norm is False: var = 1

                # height = self.height * (self.end_epoch - epoch) / (self.end_epoch - self.start_epoch)
                ## hard code
                if type(self.height) is list:
                    base_height = self.height[cur_class_id]
                else:
                    base_height = self.height

                if type(self.end_epoch) is list:
                    base_end_epoch = self.end_epoch[cur_class_id]
                else:
                    base_end_epoch = self.end_epoch

                if type(self.elongation) is list:
                    base_elongation = self.elongation[cur_class_id]
                else:
                    base_elongation = self.elongation

                if self.inverse:
                    height = base_height * (base_end_epoch - epoch) / (base_end_epoch - self.start_epoch)
                else:
                    height = base_height * max(base_end_epoch - epoch, 0) / (base_end_epoch - self.start_epoch)

                if self.fixed:  # make height to be the same for all times
                    height = base_height
                if epoch > self.cut_epoch:
                    height = 0
                # print(type(pred_sigmoid), type(threshold))
                # exit()
                if self.sm:
                    if self.oto:
                        mask = (groups[:, :, cur_class_id] > 0) & (pred_sigmoid[:, :, cur_class_id] <= self.smt)
                    else:
                        mask = (target[:, :, cur_class_id] > 0) & (pred_sigmoid[:, :, cur_class_id] <= self.smt)

                    if epoch >= self.sme:
                        curriculum_weight[:, :, cur_class_id][mask] = 0.5

                elif self.sma:
                    mask = (target[:, :, cur_class_id] > 0) & (groups[:, :, cur_class_id] <= 0) & (pred_sigmoid[:, :, cur_class_id] <= self.smt)
                    if epoch >= self.sme:
                        curriculum_weight[:, :, cur_class_id][mask] = 0.5
                else:
                    if self.oto:
                        mask = groups[:, :, cur_class_id] > 0
                    else:
                        mask = target[:, :, cur_class_id] > 0
                    # print(mask.shape)
                    # exit()
                    curriculum_weight[:, :, cur_class_id][mask] = height / (
                            1 + torch.exp(base_elongation * (pred_sigmoid[:, :, cur_class_id][mask].detach() - threshold) / var)) + 1 - height / 2
                    curriculum_weight[:, :, cur_class_id][(pred_sigmoid[:, :, cur_class_id] > threshold) & (mask)] *= self.pos_norm
                    curriculum_weight[:, :, cur_class_id][(pred_sigmoid[:, :, cur_class_id] <= threshold) & (mask)] *= self.neg_norm

            # print(torch.where(groups[:, :, cur_class_id] > 0))
            # print(torch.where(target[:, :, cur_class_id]> 0))
            # exit()

        else:
            curriculum_weight = torch.ones(pred_sigmoid.shape, device=pred_sigmoid.device)
            # print(torch.where(curriculum_weight != 1))
            # print(torch.where(target > 0))


        self.confidence_all = self.groups_confidence(pred_sigmoid, groups)

        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        if self.use_curriculum_loss:
            return loss * weights * curriculum_weight, curriculum_weight
        else:
            return loss * weights, curriculum_weight






class WeightedSmoothL1Loss(nn.Module):
    """
    Code-wise Weighted Smooth L1 Loss modified based on fvcore.nn.smooth_l1_loss
    https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/smooth_l1_loss.py
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    """
    def __init__(self, beta: float = 1.0 / 9.0, code_weights: list = None):
        """
        Args:
            beta: Scalar float.
                L1 to L2 change point.
                For beta values < 1e-5, L1 loss is computed.
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedSmoothL1Loss, self).__init__()
        self.beta = beta
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    @staticmethod
    def smooth_l1_loss(diff, beta):
        if beta < 1e-5:
            loss = torch.abs(diff)
        else:
            n = torch.abs(diff)
            loss = torch.where(n < beta, 0.5 * n ** 2 / beta, n - 0.5 * beta)

        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = self.smooth_l1_loss(diff, self.beta)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedL1Loss(nn.Module):
    def __init__(self, code_weights: list = None):
        """
        Args:
            code_weights: (#codes) float list if not None.
                Code-wise weights.
        """
        super(WeightedL1Loss, self).__init__()
        if code_weights is not None:
            self.code_weights = np.array(code_weights, dtype=np.float32)
            self.code_weights = torch.from_numpy(self.code_weights).cuda()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor = None):
        """
        Args:
            input: (B, #anchors, #codes) float tensor.
                Ecoded predicted locations of objects.
            target: (B, #anchors, #codes) float tensor.
                Regression targets.
            weights: (B, #anchors) float tensor if not None.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted smooth l1 loss without reduction.
        """
        target = torch.where(torch.isnan(target), input, target)  # ignore nan targets

        diff = input - target
        # code-wise weighting
        if self.code_weights is not None:
            diff = diff * self.code_weights.view(1, 1, -1)

        loss = torch.abs(diff)

        # anchor-wise weighting
        if weights is not None:
            assert weights.shape[0] == loss.shape[0] and weights.shape[1] == loss.shape[1]
            loss = loss * weights.unsqueeze(-1)

        return loss


class WeightedCrossEntropyLoss(nn.Module):
    """
    Transform input to fit the fomation of PyTorch offical cross entropy loss
    with anchor-wise weighting.
    """
    def __init__(self):
        super(WeightedCrossEntropyLoss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predited logits for each class.
            target: (B, #anchors, #classes) float tensor.
                One-hot classification targets.
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            loss: (B, #anchors) float tensor.
                Weighted cross entropy loss without reduction
        """
        input = input.permute(0, 2, 1)
        target = target.argmax(dim=-1)
        loss = F.cross_entropy(input, target, reduction='none') * weights
        return loss


def get_corner_loss_lidar(pred_bbox3d: torch.Tensor, gt_bbox3d: torch.Tensor):
    """
    Args:
        pred_bbox3d: (N, 7) float Tensor.
        gt_bbox3d: (N, 7) float Tensor.

    Returns:
        corner_loss: (N) float Tensor.
    """
    assert pred_bbox3d.shape[0] == gt_bbox3d.shape[0]

    pred_box_corners = box_utils.boxes_to_corners_3d(pred_bbox3d)
    gt_box_corners = box_utils.boxes_to_corners_3d(gt_bbox3d)

    gt_bbox3d_flip = gt_bbox3d.clone()
    gt_bbox3d_flip[:, 6] += np.pi
    gt_box_corners_flip = box_utils.boxes_to_corners_3d(gt_bbox3d_flip)
    # (N, 8)
    corner_dist = torch.min(torch.norm(pred_box_corners - gt_box_corners, dim=2),
                            torch.norm(pred_box_corners - gt_box_corners_flip, dim=2))
    # (N, 8)
    corner_loss = WeightedSmoothL1Loss.smooth_l1_loss(corner_dist, beta=1.0)

    return corner_loss.mean(dim=1)


def compute_fg_mask(gt_boxes2d, shape, downsample_factor=1, device=torch.device("cpu")):
    """
    Compute foreground mask for images
    Args:
        gt_boxes2d: (B, N, 4), 2D box labels
        shape: torch.Size or tuple, Foreground mask desired shape
        downsample_factor: int, Downsample factor for image
        device: torch.device, Foreground mask desired device
    Returns:
        fg_mask (shape), Foreground mask
    """
    fg_mask = torch.zeros(shape, dtype=torch.bool, device=device)

    # Set box corners
    gt_boxes2d /= downsample_factor
    gt_boxes2d[:, :, :2] = torch.floor(gt_boxes2d[:, :, :2])
    gt_boxes2d[:, :, 2:] = torch.ceil(gt_boxes2d[:, :, 2:])
    gt_boxes2d = gt_boxes2d.long()

    # Set all values within each box to True
    B, N = gt_boxes2d.shape[:2]
    for b in range(B):
        for n in range(N):
            u1, v1, u2, v2 = gt_boxes2d[b, n]
            fg_mask[b, v1:v2, u1:u2] = True

    return fg_mask


def neg_loss_curriculum(pred, gt, radius_map, box_mask, mask=None, epoch=None):
    """
    Refer to https://github.com/tianweiy/CenterPoint.
    Modified focal loss. Exactly the same as CornerNet. Runs faster and costs a little bit more memory
    Args:
        pred: (batch x c x h x w)
        gt: (batch x c x h x w)
        mask: (batch x h x w)
    Returns:
    """

    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    batch_size = pos_inds.shape[0]

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds


    num_obj = pos_inds.float().sum()
    avg_confidence = (pred * pos_inds).sum() / num_obj   # the average prediction confidence of all objects in this head

    num_objects = 0
    for batch in range(batch_size):
        nonzero_idx = torch.where(radius_map[batch][:, 3] > 0)[0]  # non-zero index of the objects in current batch
        for i in nonzero_idx:
            idx = i.item()
            cur_class_id = radius_map[batch, idx, 0].item()     # the class of current object
            center_x = radius_map[batch, idx, 1].item()         # the center x of current object
            center_y = radius_map[batch, idx, 2].item()         # the center y of current object
            center = (center_x, center_y)
            radius = radius_map[batch, idx, 3].item()           # the radius of the gaussian circle of current object

            pred_confidence = pred[batch, cur_class_id, center_y, center_x]   # the prediction confidence of current object. Pay attention to the order of x and y.
            assert pos_inds[batch, cur_class_id, center_y, center_x] == 1

            #threshold = avg_confidence.item()/2  # threshold for determining easy and hard instances
            #weight = 0.5   # weight for controlling curriculum

            threshold = avg_confidence.item() * 0.3  # threshold for determining easy and hard instances
            elongation = -10
            height = 1
            weight = height / (1 + np.exp(elongation * (pred_confidence.item() - threshold))) + 1 - height / 2
            #weight = 1.5

            #if pred_confidence.item() < threshold and epoch >= 1:
            if epoch >= 1:
            #if pred_confidence.item() > threshold and epoch >= 1:
                box_mask[batch, idx] = weight  # change the weight for bbox regression
                centernet_utils.draw_mask_to_heatmap(mask[batch, cur_class_id], center, radius, k=weight)  # draw weight mask for current object
                assert mask[batch, cur_class_id, center_y, center_x] == weight

            num_objects += 1


    #print(num_objects, num_obj)
    #assert num_objects == num_obj



    if mask is not None:
        mask = mask[:, None, :, :].float()
        pos_loss = pos_loss * mask
        neg_loss = neg_loss * mask
        num_pos = (pos_inds.float() * mask).sum()
    else:
        num_pos = pos_inds.float().sum()

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()



    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss, box_mask, avg_confidence


def neg_loss_cornernet(pred, gt, mask=None):
    """
    Refer to https://github.com/tianweiy/CenterPoint.
    Modified focal loss. Exactly the same as CornerNet. Runs faster and costs a little bit more memory
    Args:
        pred: (batch x c x h x w)
        gt: (batch x c x h x w)
        mask: (batch x h x w)
    Returns:
    """
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)

    loss = 0

    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    if mask is not None:
        mask = mask[:, None, :, :].float()
        pos_loss = pos_loss * mask
        neg_loss = neg_loss * mask
        num_pos = (pos_inds.float() * mask).sum()
    else:
        num_pos = pos_inds.float().sum()

    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    confidence = (pred*pos_inds).sum()/num_pos
    # print('Current thredhold is: ', thredhold)
    #print('pos_position', pred[gt.eq(1)])


    if num_pos == 0:
        loss = loss - neg_loss
    else:
        loss = loss - (pos_loss + neg_loss) / num_pos
    return loss, confidence


class FocalLossCenterNet(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """
    def __init__(self):
        super(FocalLossCenterNet, self).__init__()
        self.neg_loss = neg_loss_cornernet

    def forward(self, out, target, mask=None):
        return self.neg_loss(out, target, mask=mask)


class FocalLossCenterCurriculumMerge(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """

    def __init__(self, model_config, conf_shape=None):
        super(FocalLossCenterCurriculumMerge, self).__init__()
        # self.neg_loss = neg_loss_curriculum
        self.model_config = model_config
        self.curriculum = self.model_config.get('LOSS_CURRICULUM', None)
        self.use_curriculum_loss = self.curriculum.get('UCL', True)
        self.oto = self.curriculum.get('OTO', False)

        self.start_epoch = self.curriculum.get('START', 0)
        self.end_epoch = self.curriculum.get('END', 30)
        self.cut_epoch = self.curriculum.get('CUT', 10000)
        self.alpha = self.curriculum.get('ALPHA', 0.001)
        self.elongation = self.curriculum.get('ELONGATION', -10)
        self.height = self.curriculum.get('HEIGHT', 1)
        self.offset = self.curriculum.get('OFFSET', 0)
        self.inverse = self.curriculum.get('INV', False)
        self.use_norm = self.curriculum.get('NORM', False)
        self.pos_weight = self.curriculum.get('POSW', 1)

        self.fixed = self.curriculum.get('FIXED', False)

        self.merge_scores = self.curriculum.get('MERGE_SCORES', False)

        self.lift = self.curriculum.get('LIFT', 0.0)
        self.sm = self.curriculum.get('SM', False)
        self.sme = self.curriculum.get('SME', 20)
        self.smt = self.curriculum.get('SMT', 0.15)
        self.sma = self.curriculum.get('SMA', False)
        self.smw = self.curriculum.get('SMW', 0.5)

        print(self.start_epoch, self.end_epoch, self.alpha, self.elongation,
              self.height, self.offset, self.inverse, self.merge_scores)

        self.class_num = 3

        from scipy.stats import norm
        self.pos_norm = 0.5 / (1 - norm.cdf(self.offset)) * self.pos_weight
        self.neg_norm = 0.5 / norm.cdf(self.offset)

        self.means, self.stds = None, None

        self.conf_shape = conf_shape
        self.confidence_all = 0

    def update_scores(self, pred, gt):
        if self.merge_scores:
            pos_inds = gt.eq(1)
            num_obj = pos_inds.float().sum().cpu().detach().numpy()

            scores = pred[pos_inds].cpu().detach().numpy()
            # avg_c, std_c = np.mean(scores), np.std(scores)
            res = [np.sum(scores), np.sum(scores ** 2), len(scores)]
            res = torch.from_numpy(np.array(res)).cuda()

            world_size = torch.distributed.get_world_size()
            res_list = [torch.zeros_like(res) for _ in range(world_size)]
            torch.distributed.all_gather(res_list, res)
            res_list = np.array([e.cpu().numpy() for e in res_list])
            res_list = np.sum(res_list, axis=0)
            assert len(res_list) == 3

            src_sum, square_sum, nums = res_list[0], res_list[1], res_list[2]
            if nums == 0:
                return

            mean = src_sum / nums
            if (square_sum + nums * mean ** 2 - 2 * mean * src_sum) <= 0:
                std = 0
            else:
                std = np.sqrt((square_sum + nums * mean ** 2 - 2 * mean * src_sum) / nums)

            if self.means is None:
                self.means, self.stds = mean, std
            else:
                self.means = (1 - self.alpha) * self.means + self.alpha * mean
                self.stds = (1 - self.alpha) * self.stds + self.alpha * std
        else:
            self.class_num = gt.shape[1]
            if self.means is None:
                self.means = [None] * self.class_num
                self.stds = [None] * self.class_num

            for idx in range(self.class_num):
                pos_inds = gt[:, idx, :, :].eq(1)
                num_obj = pos_inds.float().sum().cpu().detach().numpy()
                scores = pred[:, idx, :, :][pos_inds].cpu().detach().numpy()

                res = [np.sum(scores), np.sum(scores ** 2), len(scores)]
                res = torch.from_numpy(np.array(res)).cuda()

                world_size = torch.distributed.get_world_size()
                res_list = [torch.zeros_like(res) for _ in range(world_size)]
                torch.distributed.all_gather(res_list, res)
                res_list = np.array([e.cpu().numpy() for e in res_list])
                res_list = np.sum(res_list, axis=0)
                assert len(res_list) == 3

                src_sum, square_sum, nums = res_list[0], res_list[1], res_list[2]
                if nums == 0:
                    return

                mean = src_sum / nums
                if (square_sum + nums * mean ** 2 - 2 * mean * src_sum) <= 0:
                    std = 0
                else:
                    std = np.sqrt((square_sum + nums * mean ** 2 - 2 * mean * src_sum) / nums)

                # print( src_sum, square_sum, nums)
                # if square_sum + nums*mean**2 - 2 * mean * src_sum < 0:
                #     exit(1)

                if self.means[idx] is None:
                    self.means[idx], self.stds[idx] = mean, std
                else:
                    self.means[idx] = (1 - self.alpha) * self.means[idx] + self.alpha * mean
                    self.stds[idx] = (1 - self.alpha) * self.stds[idx] + self.alpha * std
        # if torch.distributed.get_rank() == 0:
        #     print(self.means, self.stds)

    def forward(self, out, target, radius_map, box_mask, mask=None, epoch=None):

        return self.neg_loss(out, target, radius_map, box_mask, mask=mask, epoch=epoch)

    def group_confifence(self, pred, radius_map, group, class_id=None):
        if class_id is None:
            loc = torch.where(radius_map[:, :, -1] == group)
        else:
            loc = torch.where((radius_map[:, :, -1] == group) & (radius_map[:, :, 0] == class_id))
        batch_ind = loc[0].unsqueeze(0)

        ind = torch.cat((batch_ind, radius_map[:, :, :][loc][:, 0:3].permute(1, 0)), dim=0)

        index = (ind[0], ind[1], ind[3], ind[2])
        length = len(pred[index])
        # if length == 0:
        #     confidence = 0
        # else:
        #     confidence = (pred[index].sum()/length)

        confidence = (pred[index].sum(), length)

        return confidence

    def confidence_of_all_groups(self, pred, radius_map, device):
        confidence_all = torch.zeros(self.conf_shape).to(device)
        num_all = torch.zeros(self.conf_shape).to(device)
        for class_id in range(self.conf_shape[0]):
            for group_id in range(self.conf_shape[1]):
                group = group_id + 1
                confidence, num = self.group_confifence(pred, radius_map, group, class_id)

                confidence_all[class_id, group_id] = confidence.detach()
                num_all[class_id, group_id] = num
        return [confidence_all, num_all]

    def neg_loss(self, pred, gt, radius_map, box_mask, mask=None, epoch=None):
        """
            Refer to https://github.com/tianweiy/CenterPoint.
            Modified focal loss. Exactly the same as CornerNet. Runs faster and costs a little bit more memory
            Args:
                pred: (batch x c x h x w)
                gt: (batch x c x h x w)
                mask: (batch x h x w)
            Returns:
            """
        # print(gt.shape)
        # print(radius_map[:, :, -1])
        # print(radius_map[:, :, :3].shape)
        # print(radius_map[:, :, :3][radius_map[:, :, -1] > 0])
        # exit()

        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        batch_size = pos_inds.shape[0]

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_obj = pos_inds.float().sum()
        avg_confidence = (pred * pos_inds).sum() / num_obj

        if self.conf_shape is not None:
            device = pred.device
            self.confidence_all = self.confidence_of_all_groups(pred, radius_map, device)

        if radius_map.shape[-1] >= 5:
            confidence_true = 1
            confidence_aug = 2

        if self.use_curriculum_loss:
            true_object_gt = torch.zeros(gt.shape, device=radius_map.device)
            for batch in range(batch_size):
                true_inds = radius_map[batch, :, :3][radius_map[batch, :, -1] > 0]
                class_id = true_inds[:, 0]
                x = true_inds[:, 1]
                y = true_inds[:, 2]

                true_object_gt[batch][class_id, y, x] = 1
            # print(torch.where(true_object_gt == 1))
            # print(torch.where(gt == 1))
            # exit()

            #self.update_scores(pred, gt)
            self.update_scores(pred, true_object_gt)

            for batch in range(batch_size):
                nonzero_idx = torch.where(radius_map[batch][:, 3] > 0)[0]  # non-zero index of the objects in current batch
                true_object_idx = torch.where(radius_map[batch][:, -1] > 0)[0]
                # print(nonzero_idx)
                # print(true_object_idx)
                # print(self.oto)
                # exit()
                for i in nonzero_idx:
                    if self.oto and i not in true_object_idx:
                        #print('you need to stop here', i)
                        continue
                    idx = i.item()
                    cur_class_id = radius_map[batch, idx, 0].item()  # the class of current object
                    center_x = radius_map[batch, idx, 1].item()  # the center x of current object
                    center_y = radius_map[batch, idx, 2].item()  # the center y of current object
                    center = (center_x, center_y)
                    radius = radius_map[batch, idx, 3].item()  # the radius of the gaussian circle of current object

                    pred_confidence = pred[
                        batch, cur_class_id, center_y, center_x]  # the prediction confidence of current object. Pay attention to the order of x and y.
                    assert pos_inds[batch, cur_class_id, center_y, center_x] == 1

                    if self.merge_scores:
                        threshold = self.means + self.offset * self.stds
                        var = self.stds
                    else:
                        threshold = self.means[cur_class_id] + self.offset * self.stds[cur_class_id]
                        var = self.stds[cur_class_id]

                    if threshold is None:
                        print(f'threshold is None at epoch {epoch}')
                        threshold = 0.5
                        var = 0.2
                    if self.use_norm is False: var = 1

                    # height = self.height * (self.end_epoch - epoch) / (self.end_epoch - self.start_epoch)
                    ## hard code
                    if type(self.height) is list:
                        base_height = self.height[cur_class_id]
                    else:
                        base_height = self.height

                    if type(self.end_epoch) is list:
                        base_end_epoch = self.end_epoch[cur_class_id]
                    else:
                        base_end_epoch = self.end_epoch

                    if type(self.elongation) is list:
                        base_elongation = self.elongation[cur_class_id]
                    else:
                        base_elongation = self.elongation

                    if self.inverse:
                        height = base_height * (base_end_epoch - epoch) / (base_end_epoch - self.start_epoch)
                    else:
                        height = base_height * max(base_end_epoch - epoch, 0) / (base_end_epoch - self.start_epoch)

                    if self.fixed:  # make height to be the same for all times
                        height = base_height
                    if epoch > self.cut_epoch:
                        height = 0

                    if i in true_object_idx:
                        lift = self.lift
                    else:
                        lift = 0

                    weight = lift + height / (
                                1 + np.exp(base_elongation * (pred_confidence.item() - threshold) / var)) + 1 - height / 2



                    if (pred_confidence.item() > threshold):
                        weight *= self.pos_norm
                    else:
                        weight *= self.neg_norm

                    if self.sm:
                        if epoch >= self.sme and pred_confidence.item() <= self.smt:
                            weight = self.smw
                        else:
                            weight = 1.0
                    elif self.sma:
                        if epoch >= self.sme and i not in true_object_idx and pred_confidence.item() <= self.smt:
                            weight = self.smw
                        else:
                            weight = 1.0



                    # if pred_confidence.item() > threshold and epoch >= 1:
                    box_mask[batch, idx] = weight  # change the weight for bbox regression
                    centernet_utils.draw_mask_to_heatmap(mask[batch, cur_class_id], center, radius,
                                                         k=weight)  # draw weight mask for current object
                    assert mask[batch, cur_class_id, center_y, center_x] == weight

        if mask is not None:
            mask = mask[:, None, :, :].float()
            pos_loss = pos_loss * mask
            neg_loss = neg_loss * mask
            num_pos = (pos_inds.float() * mask).sum()
        else:
            num_pos = pos_inds.float().sum()

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        return loss, box_mask, avg_confidence.item(), confidence_true, confidence_aug



class FocalLossCenterCurriculum(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """
    def __init__(self, model_config, conf_shape=None):
        super(FocalLossCenterCurriculum, self).__init__()
        #self.neg_loss = neg_loss_curriculum
        self.model_config = model_config
        self.curriculum = self.model_config.get('LOSS_CURRICULUM', None)
        self.cur_iter = 0

        # if self.curriculum is not None:
        #     self.threshold = self.curriculum['THRESHOLD']
        #     self.elongation = self.curriculum['ELONGATION']
        #     self.height = self.curriculum['HEIGHT']
        # else:
        #     self.threshold = 1
        #     self.elongation = -10
        #     self.height = 1

        self.use_curriculum_loss = self.curriculum.get('UCL', True)
        self.fix_threshold = self.curriculum.get('FIX', False)

        self.avg_confidence = 0.0
        self.straight = self.curriculum.get('STRAIGHT', False)
        self.only_center = self.curriculum.get('CENTER', False)
        self.K = self.curriculum.get('K', 1.0)
        self.add = self.curriculum.get('ADD', 0)
        self.radius = self.curriculum.get('RADIUS', 0)
        self.conf_shape = conf_shape

        self.merge_scores = self.curriculum.get('MERGE_SCORES', False)
        self.start_epoch = self.curriculum.get('START', 0)
        self.end_epoch = self.curriculum.get('END', 30)
        self.cut_epoch = self.curriculum.get('CUT', 10000)
        self.alpha = self.curriculum.get('ALPHA', 0.001)
        self.elongation = self.curriculum.get('ELONGATION', -10)
        self.height = self.curriculum.get('HEIGHT', 1)
        self.offset = self.curriculum.get('OFFSET', 0)
        self.inverse = self.curriculum.get('INV', False)
        self.use_norm = self.curriculum.get('NORM', False)
        self.pos_weight = self.curriculum.get('POSW', 1)
        self.fixed = self.curriculum.get('FIXED', False)

        self.confidence_all = 0
        self.tuning = self.curriculum.get('TUNING', False)

        from scipy.stats import norm
        self.pos_norm = 0.5 / (1 - norm.cdf(self.offset)) * self.pos_weight
        self.neg_norm = 0.5 / norm.cdf(self.offset)

        self.means, self.stds = None, None

        self.class_num = 3

        self.threshold = 0.5

    def update_scores(self, pred, gt):
        if self.merge_scores:
            pos_inds = gt.eq(1)
            num_obj = pos_inds.float().sum().cpu().detach().numpy()

            scores = pred[pos_inds].cpu().detach().numpy()
            # avg_c, std_c = np.mean(scores), np.std(scores)
            res = [np.sum(scores), np.sum(scores ** 2), len(scores)]
            res = torch.from_numpy(np.array(res)).cuda()

            world_size = torch.distributed.get_world_size()
            res_list = [torch.zeros_like(res) for _ in range(world_size)]
            torch.distributed.all_gather(res_list, res)
            res_list = np.array([e.cpu().numpy() for e in res_list])
            res_list = np.sum(res_list, axis=0)
            assert len(res_list) == 3

            src_sum, square_sum, nums = res_list[0], res_list[1], res_list[2]
            if nums == 0:
                return

            mean = src_sum / nums
            if (square_sum + nums * mean ** 2 - 2 * mean * src_sum) <= 0:
                std = 0
            else:
                std = np.sqrt((square_sum + nums * mean ** 2 - 2 * mean * src_sum) / nums)

            if self.means is None:
                self.means, self.stds = mean, std
            else:
                self.means = (1 - self.alpha) * self.means + self.alpha * mean
                self.stds = (1 - self.alpha) * self.stds + self.alpha * std
        else:
            self.class_num = gt.shape[1]
            if self.means is None:
                self.means = [None] * self.class_num
                self.stds = [None] * self.class_num

            for idx in range(self.class_num):
                pos_inds = gt[:, idx, :, :].eq(1)
                num_obj = pos_inds.float().sum().cpu().detach().numpy()

                scores = pred[:, idx, :, :][pos_inds].cpu().detach().numpy()
                res = [np.sum(scores), np.sum(scores ** 2), len(scores)]
                res = torch.from_numpy(np.array(res)).cuda()

                world_size = torch.distributed.get_world_size()
                res_list = [torch.zeros_like(res) for _ in range(world_size)]
                torch.distributed.all_gather(res_list, res)
                res_list = np.array([e.cpu().numpy() for e in res_list])
                res_list = np.sum(res_list, axis=0)
                assert len(res_list) == 3

                src_sum, square_sum, nums = res_list[0], res_list[1], res_list[2]
                if nums == 0:
                    return

                mean = src_sum / nums
                if (square_sum + nums * mean ** 2 - 2 * mean * src_sum) <= 0:
                    std = 0
                else:
                    std = np.sqrt((square_sum + nums * mean ** 2 - 2 * mean * src_sum) / nums)

                # print( src_sum, square_sum, nums)
                # if square_sum + nums*mean**2 - 2 * mean * src_sum < 0:
                #     exit(1)

                if self.means[idx] is None:
                    self.means[idx], self.stds[idx] = mean, std
                else:
                    self.means[idx] = (1 - self.alpha) * self.means[idx] + self.alpha * mean
                    self.stds[idx] = (1 - self.alpha) * self.stds[idx] + self.alpha * std
        # if torch.distributed.get_rank() == 0:
        #     print(self.means, self.stds)

    def forward(self, out, target, radius_map, box_mask, mask=None, epoch=None):

        return self.neg_loss(out, target, radius_map, box_mask, mask=mask, epoch=epoch)

    def group_confifence(self, pred, radius_map, group, class_id=None):
        if class_id is None:
            loc = torch.where(radius_map[:, :, -1] == group)
        else:
            loc = torch.where((radius_map[:, :, -1] == group) & (radius_map[:, :, 0] == class_id))
        batch_ind = loc[0].unsqueeze(0)

        ind = torch.cat((batch_ind, radius_map[:, :, :][loc][:, 0:3].permute(1, 0)), dim=0)

        index = (ind[0], ind[1], ind[3], ind[2])
        length = len(pred[index])
        # if length == 0:
        #     confidence = 0
        # else:
        #     confidence = (pred[index].sum()/length)

        confidence = (pred[index].sum(), length)

        return confidence

    # def confidence_of_all_groups(self, pred, radius_map, device):
    #     confidence_all = torch.zeros(self.conf_shape).to(device)
    #     for class_id in range(self.conf_shape[0]):
    #         for group_id in range(self.conf_shape[1]):
    #             group = group_id + 1
    #             confidence = self.group_confifence(pred, radius_map, group, class_id)
    #             if confidence != 0:
    #                 confidence_all[class_id, group_id] = confidence.detach()
    #             else:
    #                 confidence_all[class_id, group_id] = self.confidence_all[class_id, group_id]
    #     return confidence_all


    def confidence_of_all_groups(self, pred, radius_map, device):
        confidence_all = torch.zeros(self.conf_shape).to(device)
        num_all = torch.zeros(self.conf_shape).to(device)
        for class_id in range(self.conf_shape[0]):
            for group_id in range(self.conf_shape[1]):
                group = group_id + 1
                confidence, num = self.group_confifence(pred, radius_map, group, class_id)

                confidence_all[class_id, group_id] = confidence.detach()
                num_all[class_id, group_id] = num
        return [confidence_all, num_all]

    def neg_loss(self, pred, gt, radius_map, box_mask, mask=None, epoch=None):
        """
            Refer to https://github.com/tianweiy/CenterPoint.
            Modified focal loss. Exactly the same as CornerNet. Runs faster and costs a little bit more memory
            Args:
                pred: (batch x c x h x w)
                gt: (batch x c x h x w)
                mask: (batch x h x w)
            Returns:
            """

        if self.conf_shape is not None:
            # if self.cur_iter % 4 == 0:
                #print(self.cur_iter, self.confidence_all)
            device = pred.device
            self.confidence_all = self.confidence_of_all_groups(pred, radius_map, device)

            #self.confidence_all.to(device)


        if radius_map.shape[-1] >= 5:
            # confidence_true = self.group_confifence(pred, radius_map, 1)
            # confidence_aug = self.group_confifence(pred, radius_map, 2)
            confidence_true = 1
            confidence_aug = 2



        pos_inds = gt.eq(1).float()
        neg_inds = gt.lt(1).float()

        batch_size = pos_inds.shape[0]

        neg_weights = torch.pow(1 - gt, 4)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

        num_obj = pos_inds.float().sum()
        avg_confidence = (pred * pos_inds).sum() / num_obj  # the average prediction confidence of all objects in this head

        # num_obj_class = pos_inds.float().sum(3).sum(2).sum(0) + 0.001
        # avg_confidence_class = ((pred * pos_inds).sum(3).sum(2).sum(0).detach()) / num_obj_class

        self.avg_confidence = self.alpha * avg_confidence.item() + (1-self.alpha) * self.avg_confidence

        if self.use_curriculum_loss:
            num_objects = 0
            for batch in range(batch_size):
                if radius_map.shape[-1] >= 5:
                    # nonzero_idx = torch.where((radius_map[batch][:, 3] > 0) &
                    #                           (radius_map[batch][:, 4] == 1))[0]  # non-zero index of the objects in current batch
                    #
                    # radius_idx = torch.where(radius_map[batch][:, 3] > 0)[0]
                    # to_idx = torch.where(radius_map[batch][:, 4] > 0)[0]
                    # assert radius_idx.equal(to_idx)
                    # print(radius_map[batch][:, 4])
                    # exit()
                    nonzero_idx = torch.where((radius_map[batch][:, 3] > 0) &
                                              (radius_map[batch][:, 4] > 0))[0]  # for true objects only

                    nonzero_idx = torch.where((radius_map[batch][:, 3] > 0))[0]  # for all objects

                else:
                    nonzero_idx = torch.where(radius_map[batch][:, 3] > 0)[0]  # non-zero index of the objects in current batch

                for i in nonzero_idx:
                    idx = i.item()
                    cur_class_id = radius_map[batch, idx, 0].item()  # the class of current object
                    center_x = radius_map[batch, idx, 1].item()  # the center x of current object
                    center_y = radius_map[batch, idx, 2].item()  # the center y of current object
                    center = (center_x, center_y)
                    if self.radius != 0:
                        radius = self.radius
                    else:
                        radius = radius_map[batch, idx, 3].item() + self.add  # the radius of the gaussian circle of current object

                    pred_confidence = pred[
                        batch, cur_class_id, center_y, center_x]  # the prediction confidence of current object. Pay attention to the order of x and y.
                    assert pos_inds[batch, cur_class_id, center_y, center_x] == 1

                    threshold = self.avg_confidence * self.threshold  # threshold for determining easy and hard instances
                    if self.fix_threshold:
                        threshold = self.threshold
                    elongation = self.elongation
                    height = self.height

                    if self.straight:
                        weight = self.K * (pred_confidence.item() - threshold) + 1
                    elif self.tuning:
                        # if pred_confidence.item() <= 0.15:
                        #     weight = 0.5
                        # # elif 0.5 < pred_confidence.item() <= 2:
                        # #     weight = 1.2
                        # else:
                        #     weight = 1
                        weight = 1

                    else:
                        weight = height / (1 + np.exp(elongation * (pred_confidence.item() - threshold))) + 1 - height / 2

                    if self.start_epoch <= epoch <= self.end_epoch:
                        box_mask[batch, idx] = weight  # change the weight for bbox regression
                        if self.only_center:
                            mask[batch, cur_class_id, center_y, center_x] = weight
                        else:
                            centernet_utils.draw_mask_to_heatmap(mask[batch, cur_class_id], center, radius=radius,
                                                                 k=weight)  # draw weight mask for current object
                        assert mask[batch, cur_class_id, center_y, center_x] == weight

                    num_objects += 1


        if mask is not None:
            mask = mask[:, None, :, :].float()
            pos_loss = pos_loss * mask
            neg_loss = neg_loss * mask
            num_pos = (pos_inds.float() * mask).sum()
        else:
            num_pos = pos_inds.float().sum()

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos
        return loss, box_mask, avg_confidence.item(), confidence_true, confidence_aug


def _reg_loss(regr, gt_regr, mask):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    L1 regression loss
    Args:
        regr (batch x max_objects x dim)
        gt_regr (batch x max_objects x dim)
        mask (batch x max_objects)
    Returns:
    """
    num = mask.float().sum()
    mask = mask.unsqueeze(2).expand_as(gt_regr).float()

    # isnotnan = (~ torch.isnan(gt_regr)).float() # TODO: should it be exist?
    # mask *= isnotnan

    regr = regr * mask
    gt_regr = gt_regr * mask

    loss = torch.abs(regr - gt_regr)
    loss = loss.transpose(2, 0)

    loss = torch.sum(loss, dim=2)
    loss = torch.sum(loss, dim=1)
    # else:
    #  # D x M x B
    #  loss = loss.reshape(loss.shape[0], -1)

    # loss = loss / (num + 1e-4)
    loss = loss / torch.clamp_min(num, min=1.0)
    # import pdb; pdb.set_trace()
    return loss


def _gather_feat(feat, ind, mask=None):
    dim  = feat.size(2)
    ind  = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
    feat = feat.gather(1, ind)
    if mask is not None:
        mask = mask.unsqueeze(2).expand_as(feat)
        feat = feat[mask]
        feat = feat.view(-1, dim)
    return feat


def _transpose_and_gather_feat(feat, ind):
    feat = feat.permute(0, 2, 3, 1).contiguous()
    feat = feat.view(feat.size(0), -1, feat.size(3))
    feat = _gather_feat(feat, ind)
    return feat


class RegLossCenterNet(nn.Module):
    """
    Refer to https://github.com/tianweiy/CenterPoint
    """

    def __init__(self):
        super(RegLossCenterNet, self).__init__()

    def forward(self, output, mask, ind=None, target=None):
        """
        Args:
            output: (batch x dim x h x w) or (batch x max_objects)
            mask: (batch x max_objects)
            ind: (batch x max_objects)
            target: (batch x max_objects x dim)
        Returns:
        """
        if ind is None:
            pred = output
        else:
            pred = _transpose_and_gather_feat(output, ind)
        loss = _reg_loss(pred, target, mask)
        return loss