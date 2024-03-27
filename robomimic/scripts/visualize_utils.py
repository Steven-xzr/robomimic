"""
用于在点云中添加可视化元素的辅助函数；segmentation的可视化函数。
"""

import numpy as np

import imageio
import open3d as o3d

import colorsys
import random
import matplotlib.cm as cm
from PIL import Image


def create_plane(A, B, C, D, x_range, y_range, z_range, num_points=10000):
    """
    创建一个平面的点集。

    参数:
    - A, B, C, D: 平面方程 Ax + By + Cz + D = 0 的系数。
    - x_range: x 坐标的范围 (min, max)。
    - y_range: y 坐标的范围 (min, max)。
    - num_points: 生成的点的数量。

    返回:
    - 平面上的点集，形状为 (num_points, 3)。
    """
    if C == 0 and B == 0:
        z = np.random.uniform(z_range[0], z_range[1], num_points)
        y = np.random.uniform(y_range[0], y_range[1], num_points)
        x = (-D - C * z) / A
    elif C == 0 and A == 0:
        x = np.random.uniform(x_range[0], x_range[1], num_points)
        z = np.random.uniform(z_range[0], z_range[1], num_points)
        y = (-D - A * x) / B
    else:
        x = np.random.uniform(x_range[0], x_range[1], num_points)
        y = np.random.uniform(y_range[0], y_range[1], num_points)
        z = (-D - A * x - B * y) / C

    return np.vstack((x, y, z)).T


def add_plane_to_point_cloud(pcd, plane_points):
    """
    将平面点添加到现有的点云对象中。

    参数:
    - pcd: Open3D PointCloud 对象。
    - plane_points: 要添加的平面点集，形状为 (N, 3)。

    返回:
    - 添加了平面点后的点云对象。
    """
    # 将平面点转换为Open3D格式
    plane_pcd = o3d.geometry.PointCloud()
    plane_pcd.points = o3d.utility.Vector3dVector(plane_points)

    # 将平面点云添加到原始点云
    combined_pcd = pcd + plane_pcd
    return combined_pcd


def add_sphere_to_point_cloud(
    pcd,
    sphere_center=[0.0, 0.0, 0.0],
    sphere_radius=1.0,
    sphere_color=[0.0, 0.0, 1.0],
    points_per_unit_radius=100,
):
    """
    在已有的点云中添加一个球体。

    参数:
    - pcd: 现有的点云（o3d.geometry.PointCloud对象）
    - sphere_center: 球体中心的坐标
    - sphere_radius: 球体的半径
    - sphere_color: 球体的颜色，格式为[R, G, B]
    - points_per_unit_radius: 单位半径上的点数，用于控制球体的密度
    """
    # 生成球体的点云
    num_points = int(
        points_per_unit_radius * sphere_radius**2
    )  # 根据球体的半径和单位半径上的点数确定球体点云的点数
    sphere_points = np.random.normal(size=(num_points, 3))  # 生成在单位球内的随机点
    sphere_points *= (
        sphere_radius / np.linalg.norm(sphere_points, axis=1)[:, np.newaxis]
    )  # 归一化并扩展到指定半径
    sphere_points += sphere_center  # 移动到指定中心

    # 创建球体点云
    sphere_pcd = o3d.geometry.PointCloud()
    sphere_pcd.points = o3d.utility.Vector3dVector(sphere_points)
    sphere_pcd.colors = o3d.utility.Vector3dVector(
        np.array([sphere_color] * num_points)
    )  # 设置球体点云的颜色

    # 将球体点云合并到现有点云中
    combined_pcd = pcd + sphere_pcd

    return combined_pcd


def randomize_colors(N, bright=True):
    """
    Modified from https://github.com/matterport/Mask_RCNN/blob/master/mrcnn/visualize.py#L59
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.5
    hsv = [(1.0 * i / N, 1, brightness) for i in range(N)]
    colors = np.array(list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv)))
    rstate = np.random.RandomState(seed=20)
    np.random.shuffle(colors)
    return colors


def segmentation_to_rgb(seg_im, random_colors=False):
    """
    Helper function to visualize segmentations as RGB frames.
    NOTE: assumes that geom IDs go up to 255 at most - if not,
    multiple geoms might be assigned to the same color.
    """
    # ensure all values lie within [0, 255]
    seg_im = np.mod(seg_im, 256)

    if random_colors:
        colors = randomize_colors(N=256, bright=True)
        return (255.0 * colors[seg_im]).astype(np.uint8)
    else:
        # deterministic shuffling of values to map each geom ID to a random int in [0, 255]
        rstate = np.random.RandomState(seed=8)
        inds = np.arange(256)
        rstate.shuffle(inds)

        # use @inds to map each geom ID to a color
        return (255.0 * cm.rainbow(inds[seg_im], 3)).astype(np.uint8)[..., :3]