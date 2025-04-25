import sys
import time
from math import sqrt
import autograd.numpy as np
from tqdm import tqdm
import cv2
import open3d as o3d

sys.path.append("/home/hrz/NANO-filter")
sys.path.append("D:/code/NANO-filter")
sys.path.append("./")

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


def icp_registration(source_points, target_points):
    """
    使用ICP进行点云精细配准
    """
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_points)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_points)
    threshold = 100
    transformation = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        np.identity(4),
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


def scan_to_pose(scan, map_points, angle_min, angle_max):
    lidar_points = lidar_to_point_cloud(scan, angle_min, angle_max)
    transformation = icp_registration(lidar_points, map_points)
    x, y, yaw = extract_pose(transformation)
    dx1, dy1 = x - 10, y - 10
    dx2, dy2 = x + 10, y + 10
    dx3, dy3 = x - 10, y + 10
    y1 = sqrt(dx1**2 + dy1**2)
    y2 = sqrt(dx2**2 + dy2**2)
    y3 = sqrt(dx3**2 + dy3**2)
    return x, y, y1, y2, y3


if __name__ == "__main__":
    map_path = "./data/sim/a.pgm"
    resolution = 0.05
    origin = [-10, -10, 0]
    map_points = load_map(map_path, resolution, origin)
    angle_min = -np.pi * 1.5
    angle_max = np.pi / 2
    data = np.load("./data/sim/data.npz")
    scan = data["scan"]
    scan_t = data["scan_t"]
    wheel_vel = data["wheel_vels"]
    wheel_t = data["wheel_t"]
    pos_gt = data["ground_truth"]
    time_gt = data["ground_truth_t"]
    odom = data["odom"]
    odom_t = data["odom_t"]
    scan_ = []
    wheel_vel_ = []
    odom_ = []
    for t in time_gt:
        idx = np.argmin(np.abs(scan_t - t))
        scan_.append(scan[idx])
        idx = np.argmin(np.abs(wheel_t - t))
        wheel_vel_.append(wheel_vel[idx])
        idx = np.argmin(np.abs(odom_t - t))
        odom_.append(odom[idx])
    scan = np.array(scan_)
    wheel_vel = np.array(wheel_vel_)
    odom = np.array(odom_)
    a = 0
    x0 = np.array([pos_gt[a, 0], pos_gt[a, 1], pos_gt[a, 2]])
    model = TurtleBot()
    model.x0 = x0
    filter = UKF(model)
    x_pred = []
    all_time = []
    scan_pose = []
    for i in tqdm(range(a, len(pos_gt) - 1)):
        u = odom[i + 1] - odom[i]
        y = scan_to_pose(scan[i], map_points, angle_min, angle_max)
        scan_pose.append(y[:2])
        time1 = time.time()
        filter.predict(u)
        error = (pos_gt[i][0] - y[0]) ** 2 + (pos_gt[i][1] - y[1]) ** 2
        if error < 0.16:
            filter.update(y[2:])
        time2 = time.time()
        x_pred.append(filter.x)
        all_time.append(time2 - time1)

    x_s = np.array(x_pred)
    np.save("./results/turtle_ekf.npy", x_s)
    np.save("./results/turtle_scan_pose.npy", np.array(scan_pose))
    mean_time = np.mean(all_time)
    print("solve time: ", mean_time)
