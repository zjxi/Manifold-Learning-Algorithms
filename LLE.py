
from numpy import *

''' The implementation of LLE algorithm '''

'''
 data：需要降维的矩阵
 K：k近邻算法中的超参数
 d：目标维度
'''
def lle(data, K, d):
    N = data.shape[0]   # 求数组的行数
    D = data.shape[1]   # 求数组的列数
    X = mat(data).T        # mat()方法使data数据转换为矩阵。.T 是求转置
    X_2 = sum(data ** 2, axis=1)      # 矩阵X中每列元素平方后相加
    # 行方向吧X2 复制N份，列方向上复制1份
    dis = tile(X_2, (N, 1)) - tile(array([X_2]).T, (1, N)) - 2 * dot(data, data.T)
    index = argsort(dis)            # 从小到大排序返回索引值
    neighbour = index[:, 1:K+1]     # 每个点未排序前的位置序号

    # 计算重构权值
    if K > D:
        tol = 1e-3
    else:
        tol = 0
    W = mat(zeros((K, N)))
    for ii in range(N):
        Q= X[:, neighbour[ii]]-tile(X[:, ii], (1, K))  # Q=[Xi-Zi1,....]
        C = Q.T*Q                               # 本地协方差
        C = C + eye(K) * tol * trace(C)         # (K>D时要正则化）C+的部分迭代终止误差限，防止C过小，对数据正则化，为了数据的统一
        W[:, ii] = linalg.inv(C)*mat(ones((K, 1)))
        W[:, ii] = W[:, ii] / sum(W[:, ii])     # 归一化，计算权重矩阵

    # 计算矩阵M=(I-W_).T*(I-W_)的最小d个非零特征值对应的特征向量
    W_ = zeros((N, N))
    for i in range(N):
        W_[i][neighbour[i]] = array(W[:, i].T)[0]
    W_ = mat(W_)
    I = mat(eye(N))
    M = (I-W_).T*(I-W_)
    eig, vec = linalg.eig(mat(M))
    order = argsort(eig)
    vec = vec[:, order]
    Y = vec[:, 1:d+1]
    return array(Y)
