import numpy as np

'''the implementation of isomap'''


# 获取欧氏距离
# data: 要获取欧氏距离的矩阵，大小 m * n
# return：m * m 的矩阵，第 [i, j] 个元素代表 data 中元素 i 到元素 j 的欧氏距离
def get_distance(data):
    data_count = len(data)
    mat_distance = np.zeros([data_count, data_count], np.float32)
    for idx in range(data_count):
        for sub_idx in range(data_count):
            mat_distance[idx][sub_idx] = np.linalg.norm(data[idx] - data[sub_idx])
    return mat_distance



# 使用 Dijkstra 算法获取最短路径，并更新距离矩阵
# data: 距离矩阵，大小 m * m
# src：最短路径的起始点，范围 0 到 m-1
def dijkstra(data, src):
    inf = float('inf')
    data_count = len(data)
    col_u = data[src].copy()
    dot_remain = data_count - 1
    while dot_remain > 0:
        dot_k = np.argpartition(col_u, 1)[1]
        length = data[src][dot_k]
        for idx in range(data_count):
            if data[src][idx] > length + data[dot_k][idx]:
                data[src][idx] = length + data[dot_k][idx]
                data[idx][src] = data[src][idx]
        dot_remain -= 1
        col_u[dot_k] = inf


# mds 算法的具体实现
# data：需要降维的矩阵
# target：目标维度
# return：降维后的矩阵
def mds(data, target):
    data_count = len(data)
    if target > data_count:
        target = data_count
    val_dist_i_j = 0.0
    vec_dist_i_2 = np.zeros([data_count], np.float32)
    vec_dist_j_2 = np.zeros([data_count], np.float32)
    mat_b = np.zeros([data_count, data_count], np.float32)
    mat_distance = get_distance(data)
    for idx in range(data_count):
        for sub_idx in range(data_count):
            dist_ij_2 = np.square(mat_distance[idx][sub_idx])
            val_dist_i_j += dist_ij_2
            vec_dist_i_2[idx] += dist_ij_2
            vec_dist_j_2[sub_idx] += dist_ij_2 / data_count
        vec_dist_i_2[idx] /= data_count
    val_dist_i_j /= np.square(data_count)
    for idx in range(data_count):
        for sub_idx in range(data_count):
            dist_ij_2 = np.square(mat_distance[idx][sub_idx])
            mat_b[idx][sub_idx] = -0.5 * (dist_ij_2 - vec_dist_i_2[idx] - vec_dist_j_2[sub_idx] + val_dist_i_j)
    a, v = np.linalg.eig(mat_b)
    list_idx = np.argpartition(a, target - 1)[-target:]
    a = np.diag(np.maximum(a[list_idx], 0.0))
    return np.matmul(v[:, list_idx], np.sqrt(a))


# isomap 算法的具体实现
# data：需要降维的矩阵
# target：目标维度
# k：k 近邻算法中的超参数
# return：降维后的矩阵
def isomap(data, target, k):
    inf = float('inf')
    data_count = len(data)
    if k >= data_count:
        raise ValueError('K的值最大为数据个数 - 1')
    mat_distance = get_distance(data)
    knn_map = np.ones([data_count, data_count], np.float32) * inf
    for idx in range(data_count):
        top_k = np.argpartition(mat_distance[idx], k)[:k + 1]
        knn_map[idx][top_k] = mat_distance[idx][top_k]
    for idx in range(data_count):
        dijkstra(knn_map, idx)
    return mds(data, target)
