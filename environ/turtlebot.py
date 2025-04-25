import autograd.numpy as np
from autograd import jacobian
from .model import Model
import cv2
import matplotlib.pyplot as plt
from math import sqrt
import yaml


class TurtleBot(Model):
    def __init__(
        self,
        state_outlier_flag=False,
        measurement_outlier_flag=False,
        noise_type="Gaussian",
    ):
        super().__init__(self)
        self.dim_x = 3
        self.dim_y = 3
        self.dt = 0.02294290509717218
        self.x0 = np.array([0.0, 0.0, 0.0])
        self.P0 = np.diag(np.array([0.1, 0.1, 0.1])) ** 2
        self.state_outlier_flag = state_outlier_flag
        self.measurement_outlier_flag = measurement_outlier_flag
        self.noise_type = noise_type
        self.alpha = 2.0
        self.beta = 5.0
        self.process_std = np.array([0.1] * self.dim_x)
        self.observation_std = np.array([0.1] * self.dim_y)
        self.obs_var = np.ones(self.dim_y) * 0.1
        self.Q = np.diag(self.process_std**2)
        self.R = np.diag(self.observation_std**2)
        self.map_info = self.read_map_yaml("./data/sim/a.yaml")
        self.map = cv2.rotate(
            cv2.imread(self.map_info["image"], cv2.IMREAD_GRAYSCALE),
            cv2.ROTATE_90_CLOCKWISE * 2,
        )
        self.angles = np.linspace(np.pi, -np.pi, self.dim_y)

    def f(self, x, u=None):
        return x + u
        # v = (u[0] + u[1]) / 2
        # w = (u[0] - u[1]) / 0.233
        # return np.array(
        #     [
        #         x[0] + v * np.cos(x[2]) * self.dt,
        #         x[1] + v * np.sin(x[2]) * self.dt,
        #         x[2] + w * self.dt,
        #     ]
        # )

    def h(self, x):
        # distances = np.zeros(self.dim_y)
        # cos_angles = np.cos(x[2] + self.angles)
        # sin_angles = np.sin(x[2] + self.angles)
        # for i in range(self.dim_y):
        #     distance = 0
        #     while distance < 12:
        #         distance += 5e-4
        #         laser_x = x[0] + distance * cos_angles[i]
        #         laser_y = x[1] + distance * sin_angles[i]
        #         map_x = int(
        #             (laser_x - self.map_info["origin"][0]) / self.map_info["resolution"]
        #         )
        #         map_y = int(
        #             (laser_y - self.map_info["origin"][1]) / self.map_info["resolution"]
        #         )
        #         if (
        #             map_x < 0
        #             or map_x >= self.map.shape[1]
        #             or map_y < 0
        #             or map_y >= self.map.shape[0]
        #             or self.map[map_y, map_x] == 0
        #         ):
        #             break
        #     distances[i] = distance
        # return distances
        position = np.array([[10, 10], [-10, -10], [10, -10]])
        distances = np.sqrt(np.sum((x[:2] - position) ** 2, axis=1) - 0.1)
        return distances

    def jac_h(self, x):
        epsilon = 5e-3
        """
        使用差分法计算向量值函数的 Jacobian 矩阵
        :param x: 输入向量
        :param epsilon: 差分步长
        :return: Jacobian 矩阵 (m x n)
        """
        n = len(x)  # 输入向量的维度
        f = self.h  # 假设 loss_func 返回的是一个向量
        m = len(f(x))  # 输出向量的维度
        jacobian = np.zeros((m, n))  # 初始化 Jacobian 矩阵 (m x n)
        fx = f(x)  # 计算原始函数值
        for i in range(n):
            x_i = x.copy()
            x_i[i] += epsilon  # 对第 i 个元素增加 epsilon
            fx_i = f(x_i)  # 计算 perturbed 输出
            # 计算 Jacobian 矩阵的每一列
            for j in range(m):
                jacobian[j, i] = (fx_i[j] - fx[j]) / epsilon
        return jacobian

    def visualize_lidar(self, x, scan=None):
        if scan is not None:
            distances = scan
        else:
            distances = self.h(x)
        angles = np.linspace(np.pi, -np.pi, self.dim_y)
        plt.imshow(self.map, cmap="gray")
        plt.scatter(
            (x[0] - self.map_info["origin"][0]) / self.map_info["resolution"],
            (x[1] - self.map_info["origin"][1]) / self.map_info["resolution"],
            c="red",
        )
        for angle, distance in zip(angles, distances):
            if distance < np.inf:
                laser_x = x[0] + distance * np.cos(x[2] + angle)
                laser_y = x[1] + distance * np.sin(x[2] + angle)
                plt.plot(
                    [
                        (x[0] - self.map_info["origin"][0])
                        / self.map_info["resolution"],
                        (laser_x - self.map_info["origin"][0])
                        / self.map_info["resolution"],
                    ],
                    [
                        (x[1] - self.map_info["origin"][1])
                        / self.map_info["resolution"],
                        (laser_y - self.map_info["origin"][1])
                        / self.map_info["resolution"],
                    ],
                    c="blue",
                )
        plt.show()

    def read_map_yaml(self, yaml_file):
        with open(yaml_file, "r") as f:
            return yaml.safe_load(f)

    def f_withnoise(self, x, u=None):
        if self.state_outlier_flag:
            prob = np.random.rand()
            if prob <= 0.95:
                cov = self.Q  # 95%概率使用Q
            else:
                cov = 100 * self.Q  # 5%概率使用100Q
        else:
            cov = self.Q
        return self.f(x, u) + np.random.multivariate_normal(
            mean=np.zeros(self.dim_x), cov=cov
        )

    def h_withnoise(self, x):
        if self.noise_type == "Gaussian":
            if self.measurement_outlier_flag:
                prob = np.random.rand()
                if prob <= 0.9:
                    cov = self.R  # 95%概率使用R
                else:
                    cov = 100 * self.R  # 5%概率使用100R
            else:
                cov = self.R
            return self.h(x) + np.random.multivariate_normal(
                mean=np.zeros(self.dim_y), cov=cov
            )
        elif self.noise_type == "Beta":
            noise = np.random.beta(self.alpha, self.beta, self.dim_y)
            noise = noise - np.mean(noise)
            return self.h(x) + noise
        else:
            return self.h(x) + np.random.laplace(
                loc=0, scale=self.obs_var, size=(self.dim_y,)
            )

    def jac_f(self, x_hat, u=0):
        return jacobian(lambda x: self.f(x, u))(x_hat)


# A = TurtleBot()
# robot_position = np.array([0.0, 0.0, 0])
# # A.visualize_lidar(robot_position)
# data = np.load("./data/sim/data.npz")
# scan = data["scan"]
# ground_truth = data["ground_truth"]
# ground_truth_t = data["ground_truth_t"] / 1e9
# time_intervals = np.diff(ground_truth_t)
# average_interval = np.mean(time_intervals)
# print("Average interval:", average_interval)
# print(ground_truth_t[0])
# A.visualize_lidar(robot_position, scan[0][::10])
