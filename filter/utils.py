import autograd.numpy as np
import warnings


def is_positive_semidefinite(matrix):
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Matrix is not square")

    eigenvalues = np.linalg.eigvals(matrix)
    # print('Eigenvalues: ', eigenvalues)

    if np.any(eigenvalues < 0):
        print("Eigenvalues: ", eigenvalues)
        warnings.warn("Matrix is not positive semidefinite")


def cal_mean(func, mean, var, points, type="mean"):
    """
    type : 'mean'  'cov'
    """
    sigmas = points.sigma_points(mean, var)
    first_eval = func(sigmas[0])

    if isinstance(first_eval, np.ndarray):
        sigmas_func = np.zeros((sigmas.shape[0], *first_eval.shape))
    else:
        sigmas_func = np.zeros(sigmas.shape[0])

    for i, s in enumerate(sigmas):
        sigmas_func[i] = func(s)

    if type == "mean":
        mean_func = np.tensordot(points.Wm, sigmas_func, axes=([0], [0]))
    else:
        mean_func = np.tensordot(points.Wc, sigmas_func, axes=([0], [0]))
    return mean_func


def cal_mean_mc(func, mean, var, num_samples=10):
    samples = np.random.multivariate_normal(mean, var, num_samples)
    # 计算每个样本在函数 f 上的值
    sample_values = np.apply_along_axis(func, 1, samples)
    # print('1:', sample_values.shape)
    # 计算样本值的平均值
    expectation = np.mean(sample_values, axis=0)
    # print('2:', expectation.shape)
    return expectation


def kl_divergence(mean0, cov0, mean1, cov1):
    # 计算维数
    k = mean0.shape[0]

    # 计算协方差矩阵的逆
    cov1_inv = np.linalg.inv(cov1)

    # 计算两个分布均值之差
    mean_diff = mean1 - mean0

    # 计算公式的每一部分
    term1 = np.trace(np.dot(cov1_inv, cov0))
    term2 = np.dot(np.dot(mean_diff.T, cov1_inv), mean_diff)
    term3 = np.log(np.linalg.det(cov1) / np.linalg.det(cov0))

    # 计算 KL 散度
    kl_div = 0.5 * (term1 + term2 - k + term3)

    return kl_div


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )
