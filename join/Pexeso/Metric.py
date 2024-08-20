import numpy as np


def dis_e_point(x1, x2):
    """
    计算两个点之间的欧氏距离
    """
    return np.sqrt(np.sum((x1 - x2)**2))

def dis_e_matrix(x,y):
    a = x.shape[0]
    b = y.shape[0]
    x = x.reshape((a, 1, x.shape[1]))
    y = y.reshape((1, b, y.shape[1]))
    distances = np.sqrt(np.sum((x - y) ** 2, axis=2))
    return distances
