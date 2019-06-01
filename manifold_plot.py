
import numpy
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
from sklearn import datasets
import LaplacianEigenmap as Le
import LLE

'''
    MDS, Isomap, LLE, LE流形学习算法的可视化
'''

# 最小路径的Floyd算法
def floyd(D, n_neighbors=15):
    Max = numpy.max(D) * 1000
    n1, n2 = D.shape
    k = n_neighbors
    D1 = numpy.ones((n1, n1)) * Max
    D_arg = numpy.argsort(D, axis=1)
    for i in range(n1):
        D1[i, D_arg[i, 0:k + 1]] = D[i, D_arg[i, 0:k + 1]]
    for k in range(n1):
        for i in range(n1):
            for j in range(n1):
                if D1[i, k] + D1[k, j] < D1[i, j]:
                    D1[i, j] = D1[i, k] + D1[k, j]
    return D1


def calculate_distance(x, y):
    d = numpy.sqrt(numpy.sum((x - y) ** 2))
    return d


def calculate_distance_matrix(x, y):
    d = metrics.pairwise_distances(x, y)
    return d


def cal_B(D):
    (n1, n2) = D.shape
    DD = numpy.square(D)
    Di = numpy.sum(DD, axis=1) / n1
    Dj = numpy.sum(DD, axis=0) / n1
    Dij = numpy.sum(DD) / (n1 ** 2)
    B = numpy.zeros((n1, n1))
    for i in range(n1):
        for j in range(n2):
            B[i, j] = (Dij + DD[i, j] - Di[i] - Dj[j]) / (-2)
    return B


def MDS(data, n):
    D = calculate_distance_matrix(data, data)
    B = cal_B(D)
    Be, Bv = numpy.linalg.eigh(B)
    Be_sort = numpy.argsort(-Be)
    Be = Be[Be_sort]
    Bv = Bv[:, Be_sort]
    Bez = numpy.diag(Be[0:n])
    Bvz = Bv[:, 0:n]
    Z = numpy.dot(numpy.sqrt(Bez), Bvz.T).T
    return Z


def Isomap(data, n):
    D = calculate_distance_matrix(data, data)
    D_floyd = floyd(D)
    B = cal_B(D_floyd)
    Be, Bv = numpy.linalg.eigh(B)
    Be_sort = numpy.argsort(-Be)
    Be = Be[Be_sort]
    Bv = Bv[:, Be_sort]
    Bez = numpy.diag(Be[0:n])
    Bvz = Bv[:, 0:n]
    Z = numpy.dot(numpy.sqrt(Bez), Bvz.T).T
    return Z


# 生成数据集
def generate_curve_data():
    # 设置1000points
    xx, target = datasets.samples_generator.make_s_curve(1000, random_state=9)
    return xx, target


# 利用MDS算法，绘制三维散点图
def mds_plot_3d(data, target):
    Z_MDS = MDS(data, 3)
    ax = Axes3D(plt.figure())
    ax.scatter(Z_MDS[:, 0], Z_MDS[:, 1], Z_MDS[:, 2], c=target, s=60)
    plt.title('MDS')
    plt.show()


# 利用isomap算法，绘制三维散点图
def iosmap_plot_3d(data, target):
    Z_Isomap = Iosmap(data, 3)
    ax = Axes3D(plt.figure())
    ax.scatter(Z_Isomap[:, 0], Z_Isomap[:, 1], Z_Isomap[:, 2], c=target, s=60)
    plt.title('Isomap')
    plt.show()


# 利用LE算法，绘制三维散点图
def le_plot_3d(data, target):
    Z_LE = Le.le(data, 3)
    ax = Axes3D(plt.figure())
    ax.scatter(Z_LE[:, 0], Z_LE[:, 1], Z_LE[:, 2], c=target, s=60)
    plt.title('LE')
    plt.show()


# 利用LLE算法，绘制三维散点图
def lle_plot_3d(data, target):
    Z_LLE = LLE.lle(data, 15, 3)
    ax = Axes3D(plt.figure())
    ax.scatter(Z_LLE[:, 0], Z_LLE[:, 1], Z_LLE[:, 2], c=target, s=60)
    plt.title('LLE')
    plt.show()


# 绘制4种算法的二维对比散点图
def manifold_plot_2d(data, target):
    Z_MDS = MDS(data, 2)
    Z_Isomap = Iosmap(data, 2)
    Z_LE = Le.le(data, 2)
    Z_LLE = LLE.lle(data, 15, 2)

    plt.suptitle('Manifold  Learning  Algorithms')
    plt.subplot(2, 2, 1)
    plt.title('Isomap')
    plt.scatter(Z_Isomap[:, 0], Z_Isomap[:, 1], c=target, s=60)
    plt.subplot(2, 2, 2)
    plt.title('MDS')
    plt.scatter(Z_MDS[:, 0], Z_MDS[:, 1], c=target, s=60)
    plt.subplot(2, 2, 3)
    plt.title('Laplacian Eigenmap')
    plt.scatter(Z_LE[:, 0], Z_LE[:, 1], c=target, s=60)
    plt.subplot(2, 2, 4)
    plt.title('LLE')
    plt.scatter(Z_LLE[:, 0], Z_LLE[:, 1], c=target, s=60)


def main():
    # 四种常用的流形学习算法的对比图
    # 1000_points, 15_neighbors
    data, target = generate_curve_data()

    '''绘制4种算法的二维投影对比散点图'''
    manifold_plot_2d(data, target)

    '''分别绘制四种常用的流形学习算法的3D散点图'''
    # mds_plot_3d(data, target)
    # isomap_plot_3d(data, target)
    # le_plot_3d(data, target)
    # lle_plot_3d(data, target)


if __name__ == '__main__':
    main()
