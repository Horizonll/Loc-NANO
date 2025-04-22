import sys
import time
from math import sqrt
import autograd.numpy as np
from tqdm import tqdm

sys.path.append("/home/hrz/NANO-filter")
sys.path.append("D:/code/NANO-filter")
sys.path.append("./")

from filter import EKF, UKF, PF, NANO
from environ import TurtleBot

np.random.seed(42)

if __name__ == "__main__":
    data = np.load("./data/sim/data1.npz")
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
    filter = NANO(model, n_iterations=1, init_type="iekf", iekf_max_iter=1, lr=1)
    x_pred = []
    all_time = []
    try:
        for i in tqdm(range(a, len(pos_gt) - 1)):
            u = odom[i + 1] - odom[i]
            y = model.h(pos_gt[i] + np.random.normal(0, 0.01, 3))
            x = odom[i]
            time1 = time.time()
            filter.predict(u)
            filter.update(y)
            time2 = time.time()
            x_pred.append(filter.x)
            all_time.append(time2 - time1)
            # print(sqrt(np.sum((x_pred[-1][:-1] - pos_gt[i][:-1]) ** 2)))
    finally:
        x_pred = np.array(x_pred)
        np.save("./results/turtle_ekf.npy", x_pred)
        mean_time = np.mean(all_time)
        print("solve time: ", mean_time)
