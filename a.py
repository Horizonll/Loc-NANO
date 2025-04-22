import numpy as np
import cv2
import open3d as o3d


def load_map(map_path, resolution, origin):
    """
    加载地图图像并转换为点云
    """
    # 读取地图图像
    map_img = cv2.imread(map_path, cv2.IMREAD_GRAYSCALE)
    if map_img is None:
        raise FileNotFoundError(f"地图文件 {map_path} 未找到")

    # 将地图图像转换为点云
    map_points = []
    for y in range(map_img.shape[0]):
        for x in range(map_img.shape[1]):
            if map_img[y, x] < 128:  # 黑色像素表示障碍物
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
        if r > 0:  # 忽略无效数据
            x = r * np.cos(theta)
            y = r * np.sin(theta)
            points.append([x, y, 0])
    return np.array(points)


def icp_registration(source_points, target_points):
    """
    使用ICP算法进行点云配准
    """
    source = o3d.geometry.PointCloud()
    source.points = o3d.utility.Vector3dVector(source_points)
    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(target_points)

    # 执行ICP配准
    threshold = 1.0  # 配准距离阈值
    transformation = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
    )
    return transformation.transformation


def extract_pose(transformation):
    """
    从变换矩阵中提取机器人的 x, y 和偏航角（yaw）
    """
    # 提取平移部分
    x = transformation[0, 3]
    y = transformation[1, 3]

    # 提取偏航角（yaw），假设机器人在2D平面上运动
    yaw = np.arctan2(transformation[1, 0], transformation[0, 0])

    return x, y, yaw


def visualize_point_clouds(map_points, lidar_points, transformed_points):
    """
    可视化地图点云、激光雷达点云和配准后的点云
    """
    map_cloud = o3d.geometry.PointCloud()
    map_cloud.points = o3d.utility.Vector3dVector(map_points)
    map_cloud.paint_uniform_color([0, 1, 0])  # 绿色表示地图点云

    lidar_cloud = o3d.geometry.PointCloud()
    lidar_cloud.points = o3d.utility.Vector3dVector(lidar_points)
    lidar_cloud.paint_uniform_color([1, 0, 0])  # 红色表示激光雷达点云

    transformed_cloud = o3d.geometry.PointCloud()
    transformed_cloud.points = o3d.utility.Vector3dVector(transformed_points)
    transformed_cloud.paint_uniform_color([0, 0, 1])  # 蓝色表示配准后的点云

    # 可视化
    o3d.visualization.draw_geometries([map_cloud, lidar_cloud, transformed_cloud])


# 主程序
if __name__ == "__main__":
    # 地图参数
    map_path = "./data/sim/maze.pgm"
    resolution = 0.05
    origin = [-10, -10, 0]

    # 加载地图点云
    map_points = load_map(map_path, resolution, origin)

    # 模拟激光雷达数据
    lidar_data = np.random.uniform(0.5, 10, 640)  # 假设640条线
    angle_min = -np.pi / 2
    angle_max = np.pi / 2

    # 转换激光雷达数据为点云
    lidar_points = lidar_to_point_cloud(lidar_data, angle_min, angle_max)

    # 使用ICP进行点云配准
    transformation = icp_registration(lidar_points, map_points)

    # 提取配准后的点云
    transformed_points = (transformation[:3, :3] @ lidar_points.T).T + transformation[
        :3, 3
    ]

    # 可视化点云
    visualize_point_clouds(map_points, lidar_points, transformed_points)

    # 提取机器人位姿
    x, y, yaw = extract_pose(transformation)
    print(f"机器人位置: x={x}, y={y}, 偏航角(yaw)={np.degrees(yaw)} 度")
