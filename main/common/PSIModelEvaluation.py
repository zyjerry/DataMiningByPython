"""
    描述：本类提供模型稳定性评价指标以及图形展示，主要是计算PSI指标
    作者：Jerry Zhang
    电子邮箱：zyjerry@gmail.com
"""

import numpy
import matplotlib.pyplot as plt
import math


class PSIModelEvaluation:
    """    私有属性声明    """
    # 原始待评估数据，train_score包含训练集上的预测概率，test_score包含测试集上的预测概率，两个集合长度不一定相等
    __train_score = numpy.arange(100, dtype='float32')
    __test_score = numpy.arange(100, dtype='float32')
    # 计算结果数据————横坐标，根据训练数据划分的概率临界值
    __x = numpy.arange(11, dtype='float32')
    # 计算结果数据————纵坐标，训练集数量
    __y_train = numpy.arange(10, dtype='float32')
    # 计算结果数据————纵坐标，测试集数量
    __y_test = numpy.arange(10, dtype='float32')
    # 计算最终结果数据————PSI
    __PSI = 0.00

    """    构造函数，初始化传入待评估数据，并计算其他所有数据
           参数train_score、test_score：必须是一维的numpy array
    """
    def __init__(self, train_score, test_score):
        print("PSIModelEvaluation init.................")
        if train_score.ndim != 1 or test_score.ndim != 1:
            raise TypeError("输入的原始数据不是一维数组")
        if numpy.size(train_score) < 1000 or numpy.size(test_score) < 1000:
            raise Exception("输入的数据量不够1000")

        # 初始化私有变量数据集，并排序
        self.__train_score = numpy.sort(train_score, axis=0)
        self.__test_score = numpy.sort(test_score, axis=0)

        # 把训练集按照概率排序，并按照数量等分10份，计算__x值和__y_train值
        train_score_10 = numpy.array_split(ary=self.__train_score, indices_or_sections=10, axis=0)
        self.__x[0] = 0
        i = 1
        while i < 11:
            self.__x[i] = numpy.max(train_score_10[i-1])
            self.__y_train[i-1] = 0.1
            i += 1
        self.__x[10] = 1.00
        # self.__y_train[10] = 0.1

        # 根据__x值的概率切分点，计算__y_test，并计算PSI
        i = 0
        while i < 10:
            min_v = self.__x[i]
            max_v = self.__x[i+1]
            w = numpy.where((self.__test_score > min_v) & (self.__test_score <= max_v))
            c = w[0].size
            print(c)
            self.__y_test[i] = c/self.__test_score.size
            if self.__y_test[i] == 0:
                self.__y_test[i] = 0.1
            self.__PSI += (self.__y_test[i] - self.__y_train[i]) \
                          * math.log(self.__y_test[i] / self.__y_train[i])
            i += 1

    """    当print该类对象时返回的内容    """
    def __str__(self):
        printstr = "PSIModelEvaluation类：\n训练数据规模："
        printstr += str(self.__train_score.size) + "，测试数据规模：" + str(self.__test_score.size)
        printstr += "\nPSI值：" + str(self.__PSI)
        return printstr

    """
        画图，横轴_x，纵轴直方图__y_train、__y_test
    """
    def drawgraph(self):
        name_list = numpy.around(self.__x[1:], decimals=2)
        x = list(range(10))
        total_width, n = 0.8, 2
        width = total_width / n
        plt.bar(x, self.__y_train, width=width, label='train', fc='y')
        for i in range(len(x)):
            x[i] = x[i] + width
        plt.bar(x, self.__y_test, width=width, label='test', tick_label=name_list, fc='r')
        plt.legend()
        plt.show()


"""    当直接运行本类时，打印默认初始化值，不做任何计算    """
if __name__ == "__main__":
    pme = PSIModelEvaluation()
    print(pme)
