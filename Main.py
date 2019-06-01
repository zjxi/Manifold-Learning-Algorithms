
from LaplacianEigenmap import *
import numpy as np
from PIL import Image
from mds import *
from isomap import *
import Classifier
from LLE import *
import sys


def mds_func(data):
    data_reduced = mds(data, 20)
    np.savetxt('mds_data_reduced.txt', data_reduced, '%.7e', '\t')
    sys.stdout.write('降维操作完成，低维度数据已保存到 mds_data_reduced.txt\n')
    return data_reduced


def isomap_func(data):
    data_reduced = isomap(data, 20, 15)
    np.savetxt('isomap_data_reduced.txt', data_reduced, '%.7e', '\t')
    sys.stdout.write('降维操作完成，低维度数据已保存到 isomap_data_reduced.txt\n')

    return data_reduced


def le_func(data):
    data_reduced = le(data, 20)
    np.savetxt('le_data_reduced.txt', data_reduced, '%.7e', '\t')
    sys.stdout.write('降维操作完成，低维度数据已保存到 le_data_reduced.txt\n')

    return data_reduced


def lle_func(data):
    data_reduced = lle(data, 10, 20)
    np.savetxt('lle_data_reduced.txt', data_reduced, '%.7e', '\t')
    sys.stdout.write('降维操作完成，低维度数据已保存到 lle_data_reduced.txt\n')

    return data_reduced


def main():
    print('---正在读取数据并降维---')
    data = np.empty([110, 10000], np.float32)
    for idx in range(110):
        image = Image.open('Data/s' + str(idx + 1) + '.bmp')
        data[idx] = np.reshape(image, [10000])
    file = open('Data/labels.txt')
    label = np.array(file.readline().strip('\n').split(','), np.int32)

    '''
    算法的调用
    '''
    data_reduced = mds_func(data)
    # data_reduced = isomap_func(data)
    # data_reduced = le_func(data)
    # data_reduced = lle_func(data)

    classifier = Classifier.Classifier(20)
    for repeat in range(500):
        for idx in range(110):
            if idx % 11 != 0:
                classifier.fit(data_reduced[idx], label[idx])
        sys.stdout.write('\r正在训练，已完成 %.1f%%' % (repeat * 100 / 500))
    sys.stdout.write('\r训练完毕，下面开始测试\n')
    correct_times = 0
    for idx in range(10):
        val = classifier.classify(data_reduced[idx * 11])
        print('第 %2d 次预测值：%d，真实值：%d' % (idx + 1, val, label[idx * 11]))
        if val == label[idx * 11]:
            correct_times += 1
    print('测试完毕，准确率：%.2f%%' % (correct_times * 100 / 10))


if __name__ == '__main__':
    main()

