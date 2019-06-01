import numpy as np

'''the implementation of LE'''


def knn(data_in, central_idx, k):
    central_dot, data_len = data_in[central_idx], len(data_in)
    distance_list = np.zeros([data_len], np.float)
    for i in range(data_len):
        distance_list[i] = np.linalg.norm(data_in[i] - central_dot)
    return np.argsort(distance_list)[:k], distance_list


# k: k近邻算法的超参数, 设置为15
# dim_out: 目标维度
def le(data_in, dim_out, k=15):
    data_len = len(data_in)
    weight = np.zeros([data_len, data_len], np.float)
    for i in range(data_len):
        knn_dots, distance_list = knn(data_in, i, k)
        # weight[i][knn_dots] = np.exp(-np.square(distance_list[knn_dots]) / (i + 1.))
        weight[i][knn_dots] = 1.0
    mat_d = np.diag(np.sum(weight, axis=0))
    mat_l = mat_d - weight
    val, vec = np.linalg.eig(mat_l)
    # val_no_zero = np.where(np.abs(val - 0.0) > 1e-6)[0]
    # if len(val_no_zero) >= dim_out:
    #     val, vec = val[val_no_zero], vec[val_no_zero]
    return np.reshape(vec[np.argsort(val)[:dim_out]], [data_len, dim_out])

