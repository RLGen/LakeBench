import sys
import time
import pdb

import numpy as np
import math

# import pylab
# import matplotlib.pyplot as plt
# import matplotlib as mpl
# import pandas as pd
# from matplotlib.pyplot import MultipleLocator  # 定义刻度线间隔
from scipy.spatial import distance
from scipy.fft import fft
from sklearn.decomposition import PCA, IncrementalPCA
from scipy.spatial.distance import cdist
from sklearn.neighbors import LocalOutlierFactor

from Metric import dis_e_matrix, dis_e_point

def find_outlier_lof(data, k, c=5):
    clf = LocalOutlierFactor(n_neighbors=20, contamination='auto')
    y_pred = clf.fit_predict(data)
    outliers = data[y_pred == -1]
    top_n_outliers = outliers[:c*k]
    return top_n_outliers

def find_outlier_pivots(data, k, c=5):
    '''
    :param data:数据集
    :param k:需要选择的离群点数量
    :param c:候选备胎
    :return:
    '''
    # Step 1: Find a candidate set of pivots using FFT
    data = np.array(data)
    n, d = data.shape # num, 维度
    c = 5 if n > 5*k else int(n/k)

    top_n_outliers = find_outlier_lof(data, k, c)
    if len(top_n_outliers)<k:
        top_n_outliers = data[:k]
    
    # Step 2: Compute distance matrix and perform PCA
    distances = dis_e_matrix(data, top_n_outliers)
    # distances = np.linalg.norm(data[:, None] - top_n_outliers, axis=2)
    # distance_matrix = np.zeros((n, k))
    # for i in range(k):
    #     distance_matrix[:, i] = np.linalg.norm(data - top_n_outliers[i], axis=1)

    k1 = k if k < len(top_n_outliers) else len(top_n_outliers)
    pca = PCA(n_components=k1)
    pca.fit(distances)
    components = pca.components_ # k1 * outliers num

    # Step 3: Select pivots with largest projection on each principal component
    selected_pivots = []
    all_index = set()

    projections = np.zeros((k1, len(top_n_outliers)))
    for i in range(k1):
        all_d = []
        for j in range(len(top_n_outliers)):
            index = np.argwhere(data == top_n_outliers[j])[0][0]
            projections[i][j] = np.dot(components[i], distances[index])
            all_d.append([np.abs(projections[i][j]), index])
        # selected_pivot_index = np.argmax(projections)
        all_d.sort()
        top = 0
        di = all_d[top] # d,index
        while di[1] in all_index:
            top += 1
            di = all_d[top]
        all_index.add(di[1])
        selected_pivots.append(data[di[1]])
    return np.array(selected_pivots)


def pivot_based_mapping(point, pivots):
    """
    将点映射到其与支点之间的距离上。

    参数：
    point：表示一个点的列表。
    pivot：表示支点的列表。

    返回值：
    """
    xy = []
    for p in pivots:
        xy.append(distance.euclidean(point, p))
    return xy


def pivot_based_mappings(points, pivots):
    xys = []
    for a in points:
        xys.append(pivot_based_mapping(a, pivots))
    return xys


def inverse_pivot_based_mapping(d, pivots):
    xys = []
    for i in range(len(d)):
        xy = []
        for j in range(len(pivots[i])):
              xy.append(pivots[i][j] + math.sqrt(d[i]**2 / 2))
        xys.append(xy)
    return xys

def inverse_pivot_based_mapping1(d, pivots):
    xy = []
    x = (9+d[0]**2-d[1]**2)/6
    print(x)
    xy.append(x)
    t = d[0]**2-x**2
    if t >= 0:
        y = math.sqrt(t)
    else:
        y = -1
    xy.append(y)
    return xy


def pivot_filtering(q, x, r):
    '''
    Fasle:在同一方格中
    True：可以被过滤
    '''
    for i in range(len(q)):
        if q[i] - r > x[i] or q[i] + r < x[i]:
            return True
    return False


def pivot_validation(q, x, r):
    '''
    true：可以匹配
    '''
    for i in range(len(q)):
        if q[i] + x[i] > r:
            return False
    return True


if __name__ == '__main__':

    # for i in range(5):
    #     data = np.random.randn(20000, (i+1) *100)  # 生成一组随机数据
    #     # print(data)
    #     t1 = time.time()
    #     o = find_outlier_lof(data,7)
    #     # outliers = find_outlier_pivots(data, 7)
    #     t2 = time.time()
    #     print((i+1) *100 ,":", (t2 - t1))


    k = 7  # 指定选择的离群点个数
    data = np.random.randn(20000, 300)  # 生成一组随机数据
    print(sys.getsizeof(data))

    t1 = time.time()
    outliers = find_outlier_pivots(data, k)
    print(sys.getsizeof(outliers))
    t2 = time.time()
    print(outliers)
    # print(25*(t2-t1))

    # pivots = [[0, 0], [3, 0]]
    # points = [[0, 0], [3, 0], [2,2]]
    pivots = outliers
    points = data
    t3 = time.time()
    xys = pivot_based_mappings(points, pivots)
    print(sys.getsizeof(xys))
    t4 = time.time()
    # print(25*(t4-t3))
    # print(xys)
    # xy = inverse_pivot_based_mapping1([1.82,1.9], pivots)
    # print(xy)
    ''''''

# def get_furthest_point(data, centers):
#     """
#     获取距离聚类中心最远的点
#     """
#     max_distance = -1
#     furthest_point = None
#     for point in data:
#         if point not in centers:
#             distances = [dis_e_point(point, center) for center in centers]
#             min_distance = np.min(distances)
#             if min_distance > max_distance:
#                 max_distance = min_distance
#                 furthest_point = point
#     return furthest_point

# def k_center_clustering(data, k):
#     """
#     基于FFT算法进行k-center聚类
#     """
#     centers = [data[0]]
#     for i in range(k-1):
#         furthest_point = get_furthest_point(data, centers)
#         centers.append(furthest_point)
#     clusters = [[] for _ in range(k)]
#     for point in data:
#         distances = [dis_e_point(point, center) for center in centers]
#         min_distance = np.min(distances)
#         cluster_index = np.argmin(distances)
#         clusters[cluster_index].append(point)
#     return centers, clusters

# def detect_outliers(data, k):
#     centers, clusters = k_center_clustering(data, k)
#     distances = []
#     for i in range(k):
#         for point in clusters[i]:
#             distances.append(dis_e_point(point, centers[i]))
#     median_distance = np.median(distances)
#     outliers = []
#     for i in range(k):
#         for point in clusters[i]:
#             if dis_e_point(point, centers[i]) > 3 * median_distance:
#                 outliers.append(point)
#     return outliers

# def farthest_first_traversal1(data, k):
#     '''
#     :param data:
#     :param k:
#     :return:
#     '''
#     # 随机选择第一个点
#     centers = [data[np.random.randint(data.shape[0])]]

#     # 选择剩余的 k - 1 个中心点
#     for i in range(k - 1):
#         # 计算每个点到已选中心点的距离
#         distances = cdist(data, np.array(centers))
#         # 选择距离最远的点作为新的中心点
#         farthest_index = np.argmax(np.min(distances, axis=1))
#         centers.append(data[farthest_index])

#     return centers


# def detect_outliers1(data, k):
#     # 使用 farthest-first-traversal 算法找到 k 个中心点
#     centers = farthest_first_traversal1(data, k)

#     # 计算每个点到最近的中心点的距离
#     distances = cdist(data, np.array(centers))
#     min_distances = np.min(distances, axis=1)

#     # 计算距离的中位数和标准差
#     median_distance = np.median(min_distances)
#     std_distance = np.std(min_distances)

#     # 根据距离的中位数和标准差判断离群点
#     outliers = []
#     for i, distance in enumerate(min_distances):
#         if distance > median_distance + 3 * std_distance:
#             outliers.append(i)

#     return outliers

class Circle:
    def __init__(self, x, y, r):
        self.x = x
        self.y = y
        self.r = r

def circle_intersection(c1, c2):
    d = math.sqrt((c1.x - c2.x)**2 + (c1.y - c2.y)**2)
    if d > c1.r + c2.r or d < abs(c1.r - c2.r):
        # 两圆不相交或者相切
        return []
    elif d == 0 and c1.r == c2.r:
        # 两圆重合
        return []
    else:
        a = (c1.r**2 - c2.r**2 + d**2) / (2 * d)
        h = math.sqrt(c1.r**2 - a**2)
        x2 = c1.x + a * (c2.x - c1.x) / d
        y2 = c1.y + a * (c2.y - c1.y) / d
        x3_1 = x2 + h * (c2.y - c1.y) / d
        y3_1 = y2 - h * (c2.x - c1.x) / d
        x3_2 = x2 - h * (c2.y - c1.y) / d
        y3_2 = y2 + h * (c2.x - c1.x) / d
        return [(x3_1, y3_1), (x3_2, y3_2)]

def multi_circle_intersection(circles):
    n = len(circles)
    intersection_points = []
    for i in range(n):
        for j in range(i+1, n):
            points = circle_intersection(circles[i], circles[j])
            for p in points:
                if len(intersection_points) != 0:
                    if all(math.isclose(p[0], q[0]) and math.isclose(p[1], q[1]) for q in intersection_points):
                        print(p[0],p[1])
                        # 已有相同的点，跳过
                        continue
                if all(math.isclose(p[0], c.x) and math.isclose(p[1], c.y) for c in circles):
                    # 点在所有圆心上，跳过
                    continue
                intersection_points.append(p)
    return intersection_points

def inverse1_pivot_based_mapping(d, pivots):
    circles = []
    for i in range(len(d)):
       circles.append(Circle(pivots[i][0], pivots[i][1], d[i]))
    return multi_circle_intersection(circles)