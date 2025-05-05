import sys
import time
import os
from math import sqrt
import autograd.numpy as np
from tqdm import tqdm
import cv2
import open3d as o3d

sys.path.append(os.path.abspath("./"))
from filter import EKF, UKF, PF, NANO
from environ import TurtleBot

np.random.seed(42)


def load_map(map_path, resolution, origin):
    """
    加载地图图像并转换为点云
    """
    map_img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    y_indices, x_indices = np.where(map_img < 128)
    wx = x_indices * resolution + origin[0]
    wy = (map_img.shape[0] - y_indices) * resolution + origin[1]
    map_points = np.column_stack((wx, wy, np.zeros_like(wx)))
    return map_points + np.random.normal(0, 0.5, map_points.shape)


def lidar_to_point_cloud(lidar_data, angle_min, angle_max):
    """
    将激光雷达数据转换为点云
    """
    angles = np.linspace(angle_min, angle_max, len(lidar_data))
    valid = lidar_data > 0
    r = lidar_data[valid]
    theta = angles[valid]
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    points = np.column_stack((x, y, np.zeros_like(x)))
    return points


def icp_registration(source_points, target_points, init_x=0, init_y=0, init_yaw=0):
    """
    使用ICP进行点云精细配准
    """
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_points)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_points)
    threshold = 10000
    transformation = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        np.array(
            [
                [np.cos(init_yaw), -np.sin(init_yaw), 0, init_x],
                [np.sin(init_yaw), np.cos(init_yaw), 0, init_y],
                [0, 0, 1, 0],
                [0, 0, 0, 1],
            ]
        ),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    return transformation.transformation


def extract_pose(transformation):
    """
    从变换矩阵中提取位姿
    """
    x = transformation[0, 3]
    y = transformation[1, 3]
    yaw = np.arctan2(transformation[1, 0], transformation[0, 0])
    return x, y, yaw


def scan_to_pose(scan, map_points, angle_min, angle_max, init_pose):
    lidar_points = lidar_to_point_cloud(scan, angle_min, angle_max)
    transformation = icp_registration(
        lidar_points, map_points, init_pose[0], init_pose[1], init_pose[2] + np.pi
    )
    x, y, yaw = extract_pose(transformation)
    dx1, dy1 = x - 10, y - 10
    dx2, dy2 = x + 10, y + 10
    dx3, dy3 = x - 5, y + 10
    y1 = sqrt(dx1**2 + dy1**2)
    y2 = sqrt(dx2**2 + dy2**2)
    y3 = sqrt(dx3**2 + dy3**2)
    return x, y, y1, y2, y3


def synchronize_data(time_gt, scan_t, wheel_t, odom_t, scan, wheel_vel, odom):
    """
    同步数据到统一时间戳
    """
    scan_, wheel_vel_, odom_ = [], [], []
    for t in time_gt:
        scan_.append(scan[np.argmin(np.abs(scan_t - t))])
        wheel_vel_.append(wheel_vel[np.argmin(np.abs(wheel_t - t))])
        odom_.append(odom[np.argmin(np.abs(odom_t - t))])
    return (
        np.array(scan_) + np.random.normal(0, 0.5, (len(scan_), len(scan_[0]))),
        np.array(wheel_vel_),
        np.array(odom_),
    )


def main():
    map_path = "./data/sim/map.pgm"
    resolution = 0.05
    origin = [-10, -10, 0]
    angle_min = -np.pi * 1.5
    angle_max = np.pi / 2
    map_points = load_map(map_path, resolution, origin)

    data = np.load("./data/sim/data.npz")
    scan, scan_t = data["scan"], data["scan_t"]
    wheel_vel, wheel_t = data["wheel_vels"], data["wheel_t"]
    pos_gt, time_gt = data["ground_truth"], data["ground_truth_t"]
    odom, odom_t = data["odom"], data["odom_t"]

    scan, wheel_vel, odom = synchronize_data(
        time_gt, scan_t, wheel_t, odom_t, scan, wheel_vel, odom
    )
    x0 = np.array([pos_gt[0, 0], pos_gt[0, 1], pos_gt[0, 2]])
    model = TurtleBot()
    model.x0 = x0
    filter = NANO(model, n_iterations=1, init_type="iekf", iekf_max_iter=1, lr=1)

    x_pred, all_time, scan_pose = [], [], []
    for i in tqdm(range(len(pos_gt) - 1)):
        u = odom[i + 1] - odom[i]
        y = scan_to_pose(scan[i], map_points, angle_min, angle_max, filter.x)
        # scan_pose.append(y[:2])
        start_time = time.time()
        filter.predict(u)
        filter.update(y[-3:])
        end_time = time.time()
        x_pred.append(filter.x)
        all_time.append(end_time - start_time)
        # print(
        #     sqrt(
        #         (filter.x[0] - scan_pose[-1][0]) ** 2
        #         + (filter.x[1] - scan_pose[-1][1]) ** 2
        #     )
        # )
        np.save("./results/nano.npy", np.array(x_pred))
    # np.save("./results/turtle_scan_pose.npy", np.array(scan_pose))
    print(f"{np.mean(all_time):.4f} s")


if __name__ == "__main__":
    main()
