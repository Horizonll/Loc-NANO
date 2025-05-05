import numpy as np
import cv2
import open3d as o3d


def load_map(map_path, resolution, origin):
    """
    加载地图图像并转换为点云
    """
    map_img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    map_points = []
    for y in range(map_img.shape[0]):
        for x in range(map_img.shape[1]):
            if map_img[y, x] < 128:
                wx = x * resolution + origin[0]
                wy = (map_img.shape[0] - y) * resolution + origin[1]
                map_points.append([wx, wy, 0])
    return np.array(map_points)


def lidar_to_point_cloud(lidar_data, angle_min, angle_max):
    """
    将激光雷达数据转换为点云
    """
    angles = np.linspace(angle_min, angle_max, len(lidar_data))
    points = []
    for r, theta in zip(lidar_data, angles):
        if r > 0:
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y, 0])
    return np.array(points)


def icp_registration(source_points, target_points, init_x=0, init_y=0, init_yaw=0):
    """
    使用ICP进行点云精细配准
    """
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_points)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_points)
    threshold = 80000
    # Visualize the initial estimated position of the source point cloud
    init_transformation = np.array(
        [
            [np.cos(init_yaw), -np.sin(init_yaw), 0, init_x],
            [np.sin(init_yaw), np.cos(init_yaw), 0, init_y],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    initial_transformed_points = (
        init_transformation[:3, :3] @ source_points.T
    ).T + init_transformation[:3, 3]
    initial_cloud = o3d.geometry.PointCloud()
    initial_cloud.points = o3d.utility.Vector3dVector(initial_transformed_points)
    initial_cloud.paint_uniform_color([1, 1, 0])  # Yellow for initial estimate
    o3d.visualization.draw_geometries([target, initial_cloud])
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


def visualize_point_clouds(map_points, lidar_points, transformed_points):
    """
    可视化地图点云、激光雷达点云和配准后的点云
    """
    map_cloud = o3d.geometry.PointCloud()
    map_cloud.points = o3d.utility.Vector3dVector(map_points)
    map_cloud.paint_uniform_color([0, 1, 0])
    lidar_cloud = o3d.geometry.PointCloud()
    lidar_cloud.points = o3d.utility.Vector3dVector(lidar_points)
    lidar_cloud.paint_uniform_color([1, 0, 0])
    transformed_cloud = o3d.geometry.PointCloud()
    transformed_cloud.points = o3d.utility.Vector3dVector(transformed_points)
    transformed_cloud.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([map_cloud, lidar_cloud, transformed_cloud])


if __name__ == "__main__":
    map_path = "./data/sim/map.pgm"
    resolution = 0.05
    origin = [-10, -10, 0]

    map_points = load_map(map_path, resolution, origin)

    data = np.load("./data/sim/data.npz")
    scan = data["scan"]
    scan_t = data["scan_t"]
    pos_gt = data["ground_truth"]
    time_gt = data["ground_truth_t"]
    scan_ = []
    for t in time_gt:
        idx = np.argmin(np.abs(scan_t - t))
        scan_.append(scan[idx])
    scan = np.array(scan_)
    angle_min = -np.pi * 1.5
    angle_max = np.pi / 2
    # for i in range(0, len(scan), 10):
    i = 1200
    lidar_data = scan[i] + np.random.normal(0, 0.5, (len(scan[i]),))
    lidar_points = lidar_to_point_cloud(lidar_data, angle_min, angle_max)
    print(pos_gt[i])
    transformation = icp_registration(
        lidar_points,
        map_points,
        pos_gt[i][0],
        pos_gt[i][1],
        pos_gt[i][2] + np.pi,
    )
    transformed_points = (transformation[:3, :3] @ lidar_points.T).T + transformation[
        :3, 3
    ]
    visualize_point_clouds(map_points, lidar_points, transformed_points)
    x, y, yaw = extract_pose(transformation)
    print(f"x={x:.2f}, y={y:.2f}")
    print(f"x={pos_gt[i][0]:.2f}, y={pos_gt[i][1]:.2f}")
