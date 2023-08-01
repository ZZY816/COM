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


import pickle
import time
import numpy as np
from tqdm import tqdm
# import open3d as o3d

import copy

# import cv2
# from open3d import geometry
import numba


def rotation_3d_in_axis(points,
                        angles,
                        axis=0,
                        return_mat=False,
                        clockwise=False):
    """Rotate points by angles according to axis.

    Args:
        points (np.ndarray | torch.Tensor | list | tuple ):
            Points of shape (N, M, 3).
        angles (np.ndarray | torch.Tensor | list | tuple | float):
            Vector of angles in shape (N,)
        axis (int, optional): The axis to be rotated. Defaults to 0.
        return_mat: Whether or not return the rotation matrix (transposed).
            Defaults to False.
        clockwise: Whether the rotation is clockwise. Defaults to False.

    Raises:
        ValueError: when the axis is not in range [0, 1, 2], it will
            raise value error.

    Returns:
        (torch.Tensor | np.ndarray): Rotated points in shape (N, M, 3).
    """
    batch_free = len(points.shape) == 2
    if batch_free:
        points = points[None]

    if isinstance(angles, float) or len(angles.shape) == 0:
        angles = np.full(points.shape[:1], angles)
    #         angles = torch.full(points.shape[:1], angles)

    assert len(points.shape) == 3 and len(angles.shape) == 1 \
           and points.shape[0] == angles.shape[0], f'Incorrect shape of points ' \
                                                   f'angles: {points.shape}, {angles.shape}'

    assert points.shape[-1] in [2, 3], \
        f'Points size should be 2 or 3 instead of {points.shape[-1]}'

    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if points.shape[-1] == 3:
        if axis == 1 or axis == -2:
            rot_mat_T = np.stack([
                np.stack([rot_cos, zeros, -rot_sin]),
                np.stack([zeros, ones, zeros]),
                np.stack([rot_sin, zeros, rot_cos])
            ])
        elif axis == 2 or axis == -1:
            rot_mat_T = np.stack([
                np.stack([rot_cos, rot_sin, zeros]),
                np.stack([-rot_sin, rot_cos, zeros]),
                np.stack([zeros, zeros, ones])
            ])
        elif axis == 0 or axis == -3:
            rot_mat_T = np.stack([
                np.stack([ones, zeros, zeros]),
                np.stack([zeros, rot_cos, rot_sin]),
                np.stack([zeros, -rot_sin, rot_cos])
            ])
        else:
            raise ValueError(f'axis should in range '
                             f'[-3, -2, -1, 0, 1, 2], got {axis}')
    else:
        rot_mat_T = np.stack([
            np.stack([rot_cos, rot_sin]),
            np.stack([-rot_sin, rot_cos])
        ])

    if clockwise:
        rot_mat_T = rot_mat_T.transpose(0, 1)

    if points.shape[0] == 0:
        points_new = points
    else:
        points_new = np.einsum('aij,jka->aik', points, rot_mat_T)

    if batch_free:
        points_new = points_new.squeeze(0)

    if return_mat:
        rot_mat_T = np.einsum('jka->ajk', rot_mat_T)
        if batch_free:
            rot_mat_T = rot_mat_T.squeeze(0)
        return points_new, rot_mat_T
    else:
        return points_new


@numba.njit
def _points_in_convex_polygon_3d_jit(points, polygon_surfaces, normal_vec, d,
                                     num_surfaces):
    """
    Args:
        points (np.ndarray): Input points with shape of (num_points, 3).
        polygon_surfaces (np.ndarray): Polygon surfaces with shape of
            (num_polygon, max_num_surfaces, max_num_points_of_surface, 3).
            All surfaces' normal vector must direct to internal.
            Max_num_points_of_surface must at least 3.
        normal_vec (np.ndarray): Normal vector of polygon_surfaces.
        d (int): Directions of normal vector.
        num_surfaces (np.ndarray): Number of surfaces a polygon contains
            shape of (num_polygon).

    Returns:
        np.ndarray: Result matrix with the shape of [num_points, num_polygon].
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    ret = np.ones((num_points, num_polygons), dtype=np.bool_)
    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = (
                        points[i, 0] * normal_vec[j, k, 0] +
                        points[i, 1] * normal_vec[j, k, 1] +
                        points[i, 2] * normal_vec[j, k, 2] + d[j, k])
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret


def surface_equ_3d(polygon_surfaces):
    """

    Args:
        polygon_surfaces (np.ndarray): Polygon surfaces with shape of
            [num_polygon, max_num_surfaces, max_num_points_of_surface, 3].
            All surfaces' normal vector must direct to internal.
            Max_num_points_of_surface must at least 3.

    Returns:
        tuple: normal vector and its direction.
    """
    # return [a, b, c], d in ax+by+cz+d=0
    # polygon_surfaces: [num_polygon, num_surfaces, num_points_of_polygon, 3]
    surface_vec = polygon_surfaces[:, :, :2, :] - \
                  polygon_surfaces[:, :, 1:3, :]
    # normal_vec: [..., 3]
    normal_vec = np.cross(surface_vec[:, :, 0, :], surface_vec[:, :, 1, :])
    # print(normal_vec.shape, points[..., 0, :].shape)
    # d = -np.inner(normal_vec, points[..., 0, :])
    d = np.einsum('aij, aij->ai', normal_vec, polygon_surfaces[:, :, 0, :])
    return normal_vec, -d


def corner_to_surfaces_3d(corners):
    """convert 3d box corners from corner function above to surfaces that
    normal vectors all direct to internal.

    Args:
        corners (np.ndarray): 3D box corners with shape of (N, 8, 3).

    Returns:
        np.ndarray: Surfaces with the shape of (N, 6, 4, 3).
    """
    # box_corners: [N, 8, 3], must from corner functions in this module
    surfaces = np.array([
        [corners[:, 0], corners[:, 1], corners[:, 2], corners[:, 3]],
        [corners[:, 7], corners[:, 6], corners[:, 5], corners[:, 4]],
        [corners[:, 0], corners[:, 3], corners[:, 7], corners[:, 4]],
        [corners[:, 1], corners[:, 5], corners[:, 6], corners[:, 2]],
        [corners[:, 0], corners[:, 4], corners[:, 5], corners[:, 1]],
        [corners[:, 3], corners[:, 2], corners[:, 6], corners[:, 7]],
    ]).transpose([2, 0, 1, 3])
    return surfaces


def corners_nd(dims, origin=0.5):
    """Generate relative box corners based on length per dim and origin point.

    Args:
        dims (np.ndarray, shape=[N, ndim]): Array of length per dim
        origin (list or array or float, optional): origin point relate to
            smallest point. Defaults to 0.5

    Returns:
        np.ndarray, shape=[N, 2 ** ndim, ndim]: Returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1.
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2 ** ndim), [2] * ndim),
        axis=1).astype(dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape(
        [1, 2 ** ndim, ndim])
    return corners


def points_in_convex_polygon_3d_jit(points,
                                    polygon_surfaces,
                                    num_surfaces=None):
    """Check points is in 3d convex polygons.

    Args:
        points (np.ndarray): Input points with shape of (num_points, 3).
        polygon_surfaces (np.ndarray): Polygon surfaces with shape of
            (num_polygon, max_num_surfaces, max_num_points_of_surface, 3).
            All surfaces' normal vector must direct to internal.
            Max_num_points_of_surface must at least 3.
        num_surfaces (np.ndarray, optional): Number of surfaces a polygon
            contains shape of (num_polygon). Defaults to None.

    Returns:
        np.ndarray: Result matrix with the shape of [num_points, num_polygon].
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    # num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    if num_surfaces is None:
        num_surfaces = np.full((num_polygons,), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d(polygon_surfaces[:, :, :3, :])
    # normal_vec: [num_polygon, max_num_surfaces, 3]
    # d: [num_polygon, max_num_surfaces]
    return _points_in_convex_polygon_3d_jit(points, polygon_surfaces,
                                            normal_vec, d, num_surfaces)


def center_to_corner_box3d(centers,
                           dims,
                           angles=None,
                           origin=(0.5, 1.0, 0.5),
                           axis=1):
    """Convert kitti locations, dimensions and angles to corners.

    Args:
        centers (np.ndarray): Locations in kitti label file with shape (N, 3).
        dims (np.ndarray): Dimensions in kitti label file with shape (N, 3).
        angles (np.ndarray, optional): Rotation_y in kitti label file with
            shape (N). Defaults to None.
        origin (list or array or float, optional): Origin point relate to
            smallest point. Use (0.5, 1.0, 0.5) in camera and (0.5, 0.5, 0)
            in lidar. Defaults to (0.5, 1.0, 0.5).
        axis (int, optional): Rotation axis. 1 for camera and 2 for lidar.
            Defaults to 1.

    Returns:
        np.ndarray: Corners with the shape of (N, 8, 3).
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(lwh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.reshape([-1, 1, 3])
    return corners


def boxes3d_to_corners3d_lidar(boxes3d, bottom_center=True):
    """Convert kitti center boxes to corners.

        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1

    Note:
        This function is for LiDAR boxes only.

    Args:
        boxes3d (np.ndarray): Boxes with shape of (N, 7)
            [x, y, z, x_size, y_size, z_size, ry] in LiDAR coords,
            see the definition of ry in KITTI dataset.
        bottom_center (bool, optional): Whether z is on the bottom center
            of object. Defaults to True.

    Returns:
        np.ndarray: Box corners with the shape of [N, 8, 3].
    """
    boxes_num = boxes3d.shape[0]
    x_size, y_size, z_size = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array([
        x_size / 2., -x_size / 2., -x_size / 2., x_size / 2., x_size / 2.,
        -x_size / 2., -x_size / 2., x_size / 2.], dtype=np.float32).T
    y_corners = np.array([
        -y_size / 2., -y_size / 2., y_size / 2., y_size / 2., -y_size / 2.,
        -y_size / 2., y_size / 2., y_size / 2.], dtype=np.float32).T
    if bottom_center:
        z_corners = np.zeros((boxes_num, 8), dtype=np.float32)
        z_corners[:, 4:8] = z_size.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)
    else:
        z_corners = np.array([
            -z_size / 2., -z_size / 2., -z_size / 2., -z_size / 2.,
            z_size / 2., z_size / 2., z_size / 2., z_size / 2.], dtype=np.float32).T

    ry = boxes3d[:, 6]
    zeros, ones = np.zeros(
        ry.size, dtype=np.float32), np.ones(
        ry.size, dtype=np.float32)
    rot_list = np.array([[np.cos(ry), np.sin(ry), zeros],
                         [-np.sin(ry), np.cos(ry), zeros],
                         [zeros, zeros, ones]])  # (3, 3, N)
    R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

    temp_corners = np.concatenate((x_corners.reshape(
        -1, 8, 1), y_corners.reshape(-1, 8, 1), z_corners.reshape(-1, 8, 1)),
        axis=2)  # (N, 8, 3)
    rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
    x_corners = rotated_corners[:, :, 0]
    y_corners = rotated_corners[:, :, 1]
    z_corners = rotated_corners[:, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate(
        (x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)),
        axis=2)

    return corners.astype(np.float32)

    ##### points in rbbox


def points_in_rbbox(points, rbbox, z_axis=2, origin=(0.5, 0.5, 0)):
    """Check points in rotated bbox and return indices.

    Note:
        This function is for counterclockwise boxes.

        ## ?? clockwise counter clockwise??

    Args:
        points (np.ndarray, shape=[N, 3+dim]): Points to query.
        rbbox (np.ndarray, shape=[M, 7]): Boxes3d with rotation.
        z_axis (int, optional): Indicate which axis is height.
            Defaults to 2.
        origin (tuple[int], optional): Indicate the position of
            box center. Defaults to (0.5, 0.5, 0).

    Returns:
        np.ndarray, shape=[N, M]: Indices of points in each box.
    """
    # TODO: this function is different from PointCloud3D, be careful
    # when start to use nuscene, check the input
    rbbox_corners = center_to_corner_box3d(
        rbbox[:, :3], rbbox[:, 3:6], rbbox[:, 6], origin=origin, axis=z_axis)
    surfaces = corner_to_surfaces_3d(rbbox_corners)
    indices = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
    return indices


### outside of box np ops
# rotation_3d_in_axis

def cal_occupancy(point_cnt_list, th=0):
    pos = [item > th for item in point_cnt_list]
    return sum(pos) / len(pos)


def process_ped_anno(db_info):
    """ 'name' 'path' 'sequence_name' 'sample_idx' 'gt_idx' 'box3d_lidar'
    """ 'num_points_in_gt'  'difficulty' 'global_data_offset'
    pcl_array = np.fromfile(db_info['path'], dtype=np.float32).reshape([-1, 5])
    gt_boxes_3d = copy.deepcopy(np.array([db_info['box3d_lidar']]))
    pcl_array[:, :3] += gt_boxes_3d[0, :3]
    split_part = 5
    new_gt_bboxes = np.zeros((split_part * len(gt_boxes_3d), 7))
    # print(pcl_array.shape, gt_boxes_3d)
    for i in range(len(gt_boxes_3d)):
        bottom_z = gt_boxes_3d[i][2] - gt_boxes_3d[i][5] / 2.0
        bheight = gt_boxes_3d[i][5]
        dheight = bheight / split_part
        for j in range(split_part):
            nbox = np.array([gt_boxes_3d[i][0], gt_boxes_3d[i][1], bottom_z + j * dheight + dheight / 2.0,
                             gt_boxes_3d[i][3], gt_boxes_3d[i][4], dheight, gt_boxes_3d[i][6]])
            new_gt_bboxes[split_part * i + j, :] = nbox
    # print("pre")
    indices = points_in_rbbox(pcl_array, new_gt_bboxes, z_axis=2, origin=(0.5, 0.5, 0.5))
    db_info['occupancy_array'] = indices.sum(0)
    db_info['occupancy_array'] = np.concatenate((db_info['occupancy_array'], np.zeros(7)), axis=0)
    db_info['occupancy_ratio'] = cal_occupancy(db_info['occupancy_array'])
    db_info['facade_type'] = -1
    db_info['facade_angle'] = 0


def generate_roi(box_corners):
    front_center_xy = (box_corners[0, 0, :2] + box_corners[0, 3, :2]) / 2.0
    back_center_xy = (box_corners[0, 1, :2] + box_corners[0, 2, :2]) / 2.0
    left_center_xy = (box_corners[0, 2, :2] + box_corners[0, 3, :2]) / 2.0
    right_center_xy = (box_corners[0, 0, :2] + box_corners[0, 1, :2]) / 2.0
    center_xy = (front_center_xy + back_center_xy) / 2.0
    return front_center_xy, back_center_xy, left_center_xy, right_center_xy, center_xy


def generate_center(base_xy, yaw, step):
    res_xy = [base_xy[0] + np.cos(yaw) * step,
              base_xy[1] + np.sin(yaw) * step]
    return res_xy


def process_cyc_anno(db_info):
    """ 'name' 'path' 'sequence_name' 'sample_idx' 'gt_idx' 'box3d_lidar'
    """ 'num_points_in_gt'  'difficulty' 'global_data_offset'
    pcl_array = np.fromfile(db_info['path'], dtype=np.float32).reshape([-1, 5])
    gt_boxes_3d = copy.deepcopy(np.array([db_info['box3d_lidar']]))
    pcl_array[:, :3] += gt_boxes_3d[0, :3]
    split_part = 5
    new_gt_bboxes = np.zeros((split_part * len(gt_boxes_3d), 7))
    box_corners = boxes3d_to_corners3d_lidar(gt_boxes_3d, bottom_center=True)
    # print(pcl_array.shape, gt_boxes_3d)
    for i in range(len(gt_boxes_3d)):
        front_xy, back_xy, _, _, center_xy = generate_roi(box_corners)
        assert ((center_xy[0] - gt_boxes_3d[0, 0] < 0.001))
        assert ((center_xy[1] - gt_boxes_3d[0, 1] < 0.001))
        blength = gt_boxes_3d[i][3]
        dlength = blength / (split_part * 2)
        byaw = gt_boxes_3d[i][6]
        for j in range(split_part):
            jstep = (2 * j + 1) * dlength
            nxy = generate_center(back_xy, byaw, jstep)
            nbox = np.array([nxy[0], nxy[1], gt_boxes_3d[i][2],
                             dlength * 2, gt_boxes_3d[i][4], gt_boxes_3d[i][5], gt_boxes_3d[i][6]])
            new_gt_bboxes[split_part * i + j, :] = nbox
    # print("pre")
    indices = points_in_rbbox(pcl_array, new_gt_bboxes, z_axis=2, origin=(0.5, 0.5, 0.5))
    db_info['occupancy_array'] = indices.sum(0)
    db_info['occupancy_array'] = np.concatenate((db_info['occupancy_array'], np.zeros(7)), axis=0)
    db_info['occupancy_ratio'] = cal_occupancy(db_info['occupancy_array'])
    db_info['facade_type'] = -1
    db_info['facade_angle'] = 0


def wrap_angle(angle):
    return angle % (2 * np.pi)


def determine_facade_type(gt_boxes_3d, dt=5):
    """ the vehicle visible status
        0: front_facade
        1: back_facade
        2: side_facade
        3: two facades
    """
    cx, cy = gt_boxes_3d[0, 0], gt_boxes_3d[0, 1]
    theta_0 = np.arctan2(cy, cx)
    theta_1 = gt_boxes_3d[0, 6]
    theta = wrap_angle(theta_1 - theta_0) / np.pi * 180
    if abs(theta - 180) < dt:
        return 0, theta
    if abs(theta - 0) < dt or abs(theta - 360) < dt:
        return 1, theta
    if abs(theta - 90) < dt or abs(theta - 270) < dt:
        return 2, theta
    return 3, theta


def process_vehicle_facade_anno(db_info, dt=10):
    """ 'name' 'path' 'sequence_name' 'sample_idx' 'gt_idx' 'box3d_lidar'
    """ 'num_points_in_gt'  'difficulty' 'global_data_offset'
    gt_boxes_3d = copy.deepcopy(np.array([db_info['box3d_lidar']]))
    facade_type, theta = determine_facade_type(gt_boxes_3d, dt)
    db_info['facade_type'] = facade_type
    db_info['facade_angle'] = theta


def generate_center_3d(base_xyz, yaw, step_x, step_y, step_z):
    ### in back left bottom to front right top

    ## move in length first
    len_xyz = [base_xyz[0] + np.cos(yaw) * step_x,
               base_xyz[1] + np.sin(yaw) * step_x,
               base_xyz[2]]

    ## then in width
    wid_xyz = [len_xyz[0] + np.sin(yaw) * step_y,
               len_xyz[1] - np.cos(yaw) * step_y,
               len_xyz[2]]

    ## in height
    res_xyz = [wid_xyz[0],
               wid_xyz[1],
               len_xyz[2] + step_z]
    return res_xyz



def process_vehicle_anno(db_info):
    """ 'name' 'path' 'sequence_name' 'sample_idx' 'gt_idx' 'box3d_lidar'
    """ 'num_points_in_gt'  'difficulty' 'global_data_offset'
    pcl_array = np.fromfile(db_info['path'], dtype=np.float32).reshape([-1, 5])
    gt_boxes_3d = np.array([db_info['box3d_lidar']])
    pcl_array[:,:3] += gt_boxes_3d[0,:3]
    split_x, split_y, split_z = 3, 2, 2
    split_part = split_x * split_y * split_z
    new_gt_bboxes = np.zeros((split_part * len(gt_boxes_3d), 7))
    box_corners = boxes3d_to_corners3d_lidar(gt_boxes_3d, bottom_center=True)
    for i in range(len(gt_boxes_3d)):
        front_xy, back_xy, left_xy, right_xy, center_xy = generate_roi(box_corners)
        assert((center_xy[0] - gt_boxes_3d[0, 0] < 0.001))
        assert((center_xy[1] - gt_boxes_3d[0, 1] < 0.001))

        blength = gt_boxes_3d[i][3]
        bwidth = gt_boxes_3d[i][4]
        bheight = gt_boxes_3d[i][5]
        bottom_z = gt_boxes_3d[i][2] - gt_boxes_3d[i][5] / 2.0
        dl, dw, dh = blength / split_x, bwidth / split_y, bheight / split_z
        byaw = gt_boxes_3d[i][6]
        base_xyz = [box_corners[0, 2, 0], box_corners[0, 2, 1], bottom_z]
        for z in range(split_z):
            for y in range(split_y):
                for x in range(split_x):
                    z_step = (2 * z + 1) * dh / 2.0
                    y_step = (2 * y + 1) * dw / 2.0
                    x_step = (2 * x + 1) * dl / 2.0
                    nxy = generate_center_3d(base_xyz, byaw, x_step, y_step, z_step)
                    nbox = np.array([nxy[0], nxy[1], nxy[2],
                                dl, dw, dh, gt_boxes_3d[i][6]])
                    new_gt_bboxes[z * split_x * split_y +  y * split_x + x, :] = nbox
    # print("pre")
        indices = points_in_rbbox(pcl_array, new_gt_bboxes, z_axis=2, origin=(0.5, 0.5, 0.5))
        db_info['occupancy_array'] = indices.sum(0)
        db_info['occupancy_ratio'] = cal_occupancy(db_info['occupancy_array'], th=1)


def process_vehicle_anno_for_vis(points, gt_boxes):
    import copy
    """ 'name' 'path' 'sequence_name' 'sample_idx' 'gt_idx' 'box3d_lidar'
    """ 'num_points_in_gt'  'difficulty' 'global_data_offset'
    #pcl_array = np.fromfile(db_info['path'], dtype=np.float32).reshape([-1, 5])
    pcl_array = points

    gt_boxes_3d = copy.deepcopy(gt_boxes)
    #     gt_boxes_3d[0, 3] += 0.2 * gt_boxes_3d[0, 3]
    #     gt_boxes_3d[0, 4] += 0.2 * gt_boxes_3d[0, 4]
    #     gt_boxes_3d[0, 5] += 0.2 * gt_boxes_3d[0, 5]
    #pcl_array[:, :3] += gt_boxes_3d[0, :3]

    # ori_indices = points_in_rbbox(pcl_array, gt_boxes_3d, z_axis=2, origin=(0.5, 0.5, 0.5))
    # if (ori_indices.sum() - db_info['num_points_in_gt']) > 1:
    #     print(ori_indices.sum(), db_info['num_points_in_gt'])
    split_x, split_y, split_z = 3, 2, 2
    split_part = split_x * split_y * split_z
    new_gt_bboxes = np.zeros((split_part * len(gt_boxes_3d), 7))
    box_corners = boxes3d_to_corners3d_lidar(gt_boxes_3d, bottom_center=True)
    for i in range(len(gt_boxes_3d)):
        front_xy, back_xy, left_xy, right_xy, center_xy = generate_roi(box_corners)
        assert ((center_xy[0] - gt_boxes_3d[0, 0] < 0.001))
        assert ((center_xy[1] - gt_boxes_3d[0, 1] < 0.001))

        blength = gt_boxes_3d[i][3]
        bwidth = gt_boxes_3d[i][4]
        bheight = gt_boxes_3d[i][5]
        bottom_z = gt_boxes_3d[i][2] - gt_boxes_3d[i][5] / 2.0
        dl, dw, dh = blength / split_x, bwidth / split_y, bheight / split_z
        byaw = gt_boxes_3d[i][6]
        base_xyz = [box_corners[0, 2, 0], box_corners[0, 2, 1], bottom_z]
        for z in range(split_z):
            for y in range(split_y):
                for x in range(split_x):
                    z_step = (2 * z + 1) * dh / 2.0
                    y_step = (2 * y + 1) * dw / 2.0
                    x_step = (2 * x + 1) * dl / 2.0
                    nxy = generate_center_3d(base_xyz, byaw, x_step, y_step, z_step)
                    nbox = np.array([nxy[0], nxy[1], nxy[2],
                                     dl, dw, dh, gt_boxes_3d[i][6]])
                    new_gt_bboxes[z * split_x * split_y + y * split_x + x, :] = nbox
        # print("pre")
        indices = points_in_rbbox(pcl_array, new_gt_bboxes, z_axis=2, origin=(0.5, 0.5, 0.5))

        # db_info['occupancy_array'] = indices.sum(0)
        # db_info['occupancy_ratio'] = cal_occupancy(db_info['occupancy_array'], th=1)
        # db_info['cal_num_pt'] = ori_indices.sum(0).sum()
        return new_gt_bboxes, indices.sum(0)


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
    parser.add_argument('--cfg_file', type=str, default='cfgs/waymo_models/centercurriculum_pillar_car_b2_x78_t.yaml',
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

    vis_info = db_infos['20_30'][:-1]
    show_num = 1
    distance_amplify = 0.5

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

        x1 = vis_info[i]['box3d_lidar'][0]
        y1 = vis_info[i]['box3d_lidar'][1]
        x2 = np.cos(vis_info[i]['box3d_lidar'][-1])
        y2 = np.sin(vis_info[i]['box3d_lidar'][-1])
        distance = np.sqrt(np.power(x1, 2) + np.power(y1, 2))
        angle = np.abs(x1*x2 + y1*y2)/distance
        sita = np.arccos(angle)/np.pi * 180
        #print(sita)
        all_boxes = all_boxes.cpu().numpy()

        #new_boxes, indices = process_vehicle_anno_for_vis(all_obj_points, all_boxes)
        # print(new_boxes)
        # exit()
        #print(vis_info[i])
        #print('Current_index:', i)

        V.draw_scenes(
            points=all_obj_points, gt_boxes=None
        )

    logger.info('Demo done.')





if __name__ == '__main__':
    main()
