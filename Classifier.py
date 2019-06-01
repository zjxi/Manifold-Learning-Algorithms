import numpy as np
import warnings
import os

warnings.filterwarnings("ignore")

'''构建分类器'''
class Classifier(object):
    def __init__(self, attr_count, learn_rate=0.05):
        self.__attr_count__ = attr_count
        self.__learn_rate__ = learn_rate
        self.__weight__ = np.zeros(shape=[attr_count + 1], dtype=np.float32)

    def fit(self, value, label):
        if np.shape(value) != (self.__attr_count__,):
            raise RuntimeError('初始化分类器时指定的维度为%d，但是数据的维度为%s'
                               % (self.__attr_count__, np.shape(value)))
        value = np.append(value, [1.0])
        linear_result = np.dot(value, self.__weight__)
        sigmoid_result = 1.0 / (np.exp(-linear_result) + 1.0)
        for idx in range(self.__attr_count__ + 1):
            update_val = (sigmoid_result - label) * value[idx]
            self.__weight__[idx] -= self.__learn_rate__ * update_val

    def classify(self, value):
        if np.shape(value) != (self.__attr_count__,):
            raise RuntimeError('初始化分类器时指定的维度为%d，但是数据的维度为%s'
                               % (self.__attr_count__, np.shape(value)))
        value = np.append(value, [1.0])
        linear_result = np.dot(value, self.__weight__)
        if (1.0 / (np.exp(-linear_result) + 1.0)) > 0.5:
            return 1
        else:
            return 0

    def save(self, file_name='weight.bin'):
        np.save(file_name, self.__weight__)

    def load(self, file_name='weight.bin'):
        if os.path.exists(file_name):
            self.__weight__ = np.load(file_name)
        else:
            raise RuntimeError('权重文件不存在！')
