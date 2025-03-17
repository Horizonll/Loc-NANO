import autograd.numpy as np
from autograd import jacobian
from .model import Model
import cv2
import matplotlib.pyplot as plt


class TurtleBot(Model):
    def __init__(
        self,
        state_outlier_flag=False,
        measurement_outlier_flag=False,
        noise_type="Gaussian",
    ):
        super().__init__(self)
        self.dim_x = 3
        self.dim_y = 160
        self.dt = 1.0 / 15
        self.x0 = np.array([0.0, 0.0, 0.0])
        self.P0 = np.diag(np.array([0.0001, 0.0001, 0.0001])) ** 2
        self.state_outlier_flag = state_outlier_flag
        self.measurement_outlier_flag = measurement_outlier_flag
        self.noise_type = noise_type
        self.alpha = 2.0
        self.beta = 5.0
        self.process_std = np.array([0.01] * self.dim_x)
        self.observation_std = np.array([0.01] * self.dim_y)
        self.obs_var = np.ones(self.dim_y) * 0.01
        self.Q = np.diag(self.process_std**2)
        self.R = np.diag(self.observation_std**2)
        self.map_info = self.read_map_yaml("./data/sim/map.yaml")
        self.map = cv2.rotate(
            cv2.imread(self.map_info["image"], cv2.IMREAD_GRAYSCALE),
            cv2.ROTATE_90_COUNTERCLOCKWISE,
        )

    def f(self, x, u=None):
        if u is None:
            return x
        v = (u[0] + u[1]) / 2
        w = (u[0] - u[1]) / 0.233
        return np.array(
            [
                x[0] + v * np.cos(x[2]) * self.dt,
                x[1] + v * np.sin(x[2]) * self.dt,
                x[2] + w * self.dt,
            ]
        )

    def h(self, x):
        angles = np.linspace(-np.pi, np.pi, 160)
        distances = np.zeros(160)
        resolution = self.map_info["resolution"]
        origin = self.map_info["origin"]
        map_height, map_width = self.map.shape
        for i, angle in enumerate(angles):
            distance = 0
            hit = False
            while not hit:
                distance += resolution
                laser_x = x[0] + distance * np.cos(x[2] + angle)
                laser_y = x[1] + distance * np.sin(x[2] + angle)
                try:
                    laser_x, laser_y = laser_x._value, laser_y._value
                except:
                    pass
                map_x = int((laser_x - origin[0]) / resolution)
                map_y = int((laser_y - origin[1]) / resolution)
                if map_x < 0 or map_x >= map_width or map_y < 0 or map_y >= map_height:
                    hit = True
                    distance = np.inf
                elif self.map[map_y, map_x] == 0:
                    hit = True
            distances[i] = distance
        return distances

    def visualize_lidar(self, x, scan=None):
        if scan is not None:
            distances = scan
        else:
            distances = self.h(x)
        angles = np.linspace(-np.pi, np.pi, 640)
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
            lines = f.readlines()
            map_info = {}
            for line in lines:
                key, value = line.strip().split(": ")
                if key == "image":
                    map_info[key] = value
                elif key == "resolution":
                    map_info[key] = float(value)
                elif key == "origin":
                    map_info[key] = [float(x) for x in value.strip("[]").split(", ")]
        return map_info

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

    def jac_h(self, x_hat, u=0):
        return jacobian(lambda x: self.h(x))(x_hat)


# A = TurtleBot()
# robot_position = np.array([0.0, 0.0, 0.0])
# # A.visualize_lidar(robot_position)
# data = np.load("./data/sim/data.npz")
# scan = data["scan"]
# A.visualize_lidar(robot_position, scan=None)
