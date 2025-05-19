import sys
import os
import numpy as np
from tqdm import tqdm
import cv2
import open3d as o3d
from math import sqrt

sys.path.append(os.path.abspath("./"))
from filter import EKF, UKF, PF, NANO
from environ import TurtleBot

np.random.seed(42)
map_path = "./data/sim/map.pgm"
resolution = 0.05
origin = [-10, -10, 0]
map = cv2.rotate(
    cv2.imread(map_path, cv2.IMREAD_GRAYSCALE),
    cv2.ROTATE_90_CLOCKWISE * 2,
)


def load_map(map_path, resolution, origin):
    """
    加载地图图像并转换为点云
    """
    map_img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    y_indices, x_indices = np.where(map_img < 128)
    wx = x_indices * resolution + origin[0]
    wy = (map_img.shape[0] - y_indices) * resolution + origin[1]
    map_points = np.column_stack((wx, wy, np.zeros_like(wx)))
    return map_points


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
    dx2, dy2 = x + 10, y - 10
    dx3, dy3 = x + 10, y + 10
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
        np.array(scan_) + np.random.normal(0, 1, (len(scan_), len(scan_[0]))),
        np.array(wheel_vel_),
        np.array(odom_),
    )


def get_landmarks_and_y(x, y, yaw, _scan, num_landmarks):
    landmarks = []
    angles = np.linspace(np.pi, -np.pi, 640)
    landmarks_indices = np.random.choice(len(angles), num_landmarks, replace=False)
    angles = angles[landmarks_indices]
    scan = _scan[landmarks_indices]
    cos_angles = np.cos(yaw + angles)
    sin_angles = np.sin(yaw + angles)
    for i in range(num_landmarks):
        for distance in np.arange(0, 12, 0.05):
            laser_x = x + distance * cos_angles[i]
            laser_y = y + distance * sin_angles[i]
            map_x = int((laser_x - origin[0]) / resolution)
            map_y = int((laser_y - origin[1]) / resolution)
            if (
                map[map_y, map_x] == 0
                or map_x < 0
                or map_x >= map.shape[1]
                or map_y < 0
                or map_y >= map.shape[0]
            ):
                landmarks.append([laser_x, laser_y])
                break
    if len(landmarks) < num_landmarks:
        return get_landmarks_and_y(x, y, yaw, _scan, num_landmarks)
    return landmarks, scan


def main(filter_type="ukf"):
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
    if filter_type == "ekf":
        filter = EKF(model)
    elif filter_type == "pf":
        filter = PF(model)
    elif filter_type == "ukf":
        filter = UKF(model)
    elif filter_type == "nano":
        filter = NANO(model, n_iterations=2, init_type="iekf", iekf_max_iter=2, lr=0.5)

    x_pred = []
    scan_pose_ = []
    for i in tqdm(range(len(pos_gt) - 1)):
        u = odom[i + 1] - odom[i]
        dx = u[0] * np.cos(odom[i][2]) + u[1] * np.sin(odom[i][2])
        dy = -u[0] * np.sin(odom[i][2]) + u[1] * np.cos(odom[i][2])
        u[0] = dx * np.cos(filter.x[2]) - dy * np.sin(filter.x[2])
        u[1] = dx * np.sin(filter.x[2]) + dy * np.cos(filter.x[2])
        scan_pose = scan_to_pose(scan[i], map_points, angle_min, angle_max, filter.x)
        y = scan_pose[-3:]
        filter.predict(u)
        filter.update(y)
        x_pred.append(filter.x)
        scan_pose_.append(scan_pose[:2])
    np.save(f"./results/{filter_type}.npy", np.array(x_pred))
    np.save("./results/scan_pose.npy", np.array(scan_pose_))


if __name__ == "__main__":
    if not os.path.exists("./results"):
        os.makedirs("./results")
    main("ekf")
    main("ukf")
    # main("pf")
    main("nano")
