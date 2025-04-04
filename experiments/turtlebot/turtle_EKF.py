import sys
import time
from math import sqrt
import autograd.numpy as np
from tqdm import tqdm

sys.path.append("/home/hrz/NANO-filter")
sys.path.append("D:/code/NANO-filter")
sys.path.append("./")

from filter import NANO, EKF, UKF
from environ import TurtleBot

np.random.seed(42)

if __name__ == "__main__":
    data = np.load("./data/sim/data.npz")
    scan = data["scan"]
    scan_t = data["scan_t"] / 1e9
    wheel_vel = data["wheel_vels"]
    wheel_t = data["wheel_t"] / 1e9
    pos_gt = data["ground_truth"]
    time_gt = data["ground_truth_t"] / 1e9
    odom = data["odom"]
    odom_t = data["odom_t"] / 1e9
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

    x0 = np.array([pos_gt[2000, 0], pos_gt[2000, 1], pos_gt[2000, 2]])

    model = TurtleBot()
    model.x0 = x0
    filter = EKF(model)

    x_pred = []
    all_time = []
    # 24651
    try:
        for i in tqdm(range(2000, 4000)):
            u = wheel_vel[i]
            y = scan[i][::80]
            x = odom[i]
            time1 = time.time()
            filter.predict(x)
            filter.update(y)
            time2 = time.time()
            x_pred.append(filter.x)
            all_time.append(time2 - time1)
            print(
                sqrt(np.sum((x_pred[-1] - pos_gt[i]) ** 2))
                - sqrt(np.sum((x - pos_gt[i]) ** 2))
            )
    finally:
        x_pred = np.array(x_pred)
        mean_time = np.mean(all_time)
        np.save("./results/turtle_ekf.npy", x_pred)
        print("solve time: ", mean_time)
