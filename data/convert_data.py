import rclpy
import numpy as np
from rclpy.serialization import deserialize_message
from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
from rosidl_runtime_py import message_to_ordereddict
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from irobot_create_msgs.msg import WheelVels
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped


def main():
    rclpy.init()

    # 配置存储选项
    storage_options = StorageOptions(
        uri="./data/sim/data.db3",  # 替换为你的 bag 文件路径
        storage_id="sqlite3",  # 通常使用 sqlite3 存储
    )

    # 配置转换选项
    converter_options = ConverterOptions(
        input_serialization_format="cdr", output_serialization_format="cdr"
    )

    reader = SequentialReader()
    reader.open(storage_options, converter_options)

    # 获取所有的话题信息
    topics = reader.get_all_topics_and_types()
    for topic in topics:
        print(f"Topic: {topic.name}, Type: {topic.type}")

    scan = []
    scan_t = []
    ground_truth = []
    ground_truth_t = []
    odom = []
    odom_t = []
    wheel_vels = []
    wheel_t = []
    while reader.has_next():
        (topic, data, t) = reader.read_next()
        # 根据话题类型反序列化消息
        print(f"Time: {t}, Topic: {topic}")
        if topic == "/Tracker0/pose":
            msg = deserialize_message(data, PoseStamped)
            ground_truth_t.append(t)
            q = msg.pose.orientation
            yaw = np.arctan2(
                2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )
            ground_truth.append([msg.pose.position.x, msg.pose.position.y, yaw])

        if topic == "/scan":
            msg = deserialize_message(data, LaserScan)
            scan_t.append(t)
            scan.append(msg.ranges)

        if topic == "/odom":
            msg = deserialize_message(data, Odometry)
            odom_t.append(t)
            q = msg.pose.pose.orientation
            yaw = np.arctan2(
                2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )
            odom.append([msg.pose.pose.position.x, msg.pose.pose.position.y, yaw])
        if topic == "/wheel_vels":
            msg = deserialize_message(data, WheelVels)
            wheel_t.append(t)
            wheel_vels.append(
                [
                    msg.velocity_left,
                    msg.velocity_right,
                ]
            )
        # if topic == "/amcl_pose":
        #     msg = deserialize_message(data, PoseWithCovarianceStamped)
        #     amcl.append([msg.pose.pose.position.x, msg.pose.pose.position.y])
        if topic == "/sim_ground_truth_pose":
            msg = deserialize_message(data, Odometry())
            ground_truth_t.append(t)
            q = msg.pose.pose.orientation
            yaw = np.arctan2(
                2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            )
            ground_truth.append(
                [msg.pose.pose.position.x, msg.pose.pose.position.y, yaw]
            )
    scan = np.array(scan)
    scan_t = np.array(scan_t)
    ground_truth = np.array(ground_truth)
    ground_truth_t = np.array(ground_truth_t)
    odom = np.array(odom)
    odom_t = np.array(odom_t)
    wheel_vels = np.array(wheel_vels)
    wheel_t = np.array(wheel_t)
    # amcl = np.array(amcl)
    all_data = {
        "scan": scan,
        "scan_t": scan_t,
        "ground_truth": ground_truth,
        "ground_truth_t": ground_truth_t,
        "odom": odom,
        "odom_t": odom_t,
        "wheel_vels": wheel_vels,
        "wheel_t": wheel_t,
    }
    np.savez("./data/sim/data.npz", **all_data)
    rclpy.shutdown()


if __name__ == "__main__":
    # main()
    data = np.load("./data/sim/data.npz")
    scan = data["scan"]
    scan_t = data["scan_t"]
    ground_truth = data["ground_truth"]
    ground_truth_t = data["ground_truth_t"]
    odom = data["odom"]
    odom_t = data["odom_t"]
    wheel_vels = data["wheel_vels"]
    wheel_t = data["wheel_t"]
    # print(scan.shape)
    # print(scan_t.shape)
    # print(ground_truth.shape)
    # print(ground_truth_t.shape)
    # print(odom.shape)
    # print(odom_t.shape)
    # print(wheel_vels.shape)
    # print(wheel_t.shape)
    print(scan[0])