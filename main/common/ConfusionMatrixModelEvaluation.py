"""
    描述：本类提供基于混淆矩阵的各项模型评价指标以及图形展示
    作者：Jerry Zhang
    电子邮箱：zyjerry@gmail.com
"""

import numpy
import collections
import matplotlib.pyplot as plt

class ConfusionMatrixModelEvaluation:
    """    私有属性声明    """
    # 原始待评估数据，应为n行*2列数组list，第一列预测概率y_score，第二列实际值y_true，为分类数据0/1，且1表示真0表示假
    __data = numpy.full((100, 2), 0.99)
    __y_score = numpy.arange(100)
    __y_true = numpy.arange(100)
    # 临界概率
    __probability = 0.00
    # 临界比例
    __percentage = 0.00
    # 实际为真
    __P = 0
    # 实际为假N
    __N = 0
    # 真正
    __TP = 0
    # 假正
    __FP = 0
    # 假负
    __FN = 0
    # 真负
    __TN = 0
    # 真正率/灵敏度/覆盖率/召回率/查全率 = TP/P
    __recall = 0.00
    # 命中率/查准率/精确率 = TP/(TP+FP)
    __precision = 0.00
    # 真负率/负例的覆盖率/特指度 = TN/N
    __specificity = 0.00
    # 负例的命中率 = TN/(FN+TN)
    __npv = 0.00
    # 假负率 = FN/P
    __fnr = 0.00
    # 假正率 = FP/N = 1 - specificity
    __fpr = 0.00
    # 准确性 = (TP+TN)/(P+N)
    __accuracy = 0.00
    # 误分率= (FP+FN)/(P+N) = 1 – accuracy
    __errorRate = 0.00
    # F-measure = 2/(1/Precision + 1/recall)，
    __f1 = 0.00

    """    构造函数，初始化传入待评估数据，对概率列排序，计算__P和__N
           参数data：必须是n*2的numpy array，第一列预测概率y_score，第二列实际分类值y_true    """
    def __init__(self, data):
        print("ConfusionMatrixModelEvaluation init.................")
        if data.ndim != 2 or data.shape[1] != 2:
            raise TypeError("输入的原始数据不是二维矩阵")
        if numpy.size(data)/2 < 100:
            raise Exception("输入的数据量不够")
        self.__data = data
        self.__y_score = data[:, 0]
        self.__y_true = data[:, 1]
        # 这里防止正例数量为0，后续作为分母是会发生异常，赋予最小值1
        if numpy.sum(data[:, 1]) > 0:
            self.__P = numpy.sum(data[:, 1])
        else:
            self.__P = 1
        self.__N = numpy.size(data[:, 1]) - self.__P

    """    当print该类对象时返回的内容    """
    def __str__(self):
        printstr = "ConfusionMatrixBasedModelEvaluation类：\n初始数据规模："
        printstr += str(self.__data.shape[0])
        printstr += "\n原始值：\n    正例数量：" + str(self.__P) + "，负例数量：" + str(self.__N)
        printstr += "\n    预测概率：最大值" + str(self.__data[:, 0].max()) + ",最小值"
        printstr += str(self.__data[:, 0].min()) + "，均值" + str(self.__data[:, 0].mean())
        return printstr

    """    指定计算临界概率时的各项指标    """
    def calculateindicatorsbyprobability(self, probability):
        if probability < 0 or probability > 1:
            raise TypeError("概率值必须输入数字且应介于0和1之间")
        __probability = probability
        index_greater_than_prob = numpy.where(self.__y_score >= probability)[0]
        index_less_than_prob = numpy.where(self.__y_score < probability)[0]
        index_positive = numpy.where(self.__y_true == 1)[0]
        index_negative = numpy.where(self.__y_true == 0)[0]
        self.__TP = len(set(index_greater_than_prob).intersection(set(index_positive)))
        self.__FP = len(set(index_greater_than_prob).intersection(set(index_negative)))
        self.__FN = len(set(index_less_than_prob).intersection(set(index_positive)))
        self.__TN = len(set(index_less_than_prob).intersection(set(index_negative)))
        # 真正率/灵敏度/覆盖率/召回率/查全率 = TP/P
        self.__recall = self.__TP/self.__P
        # 命中率/查准率/精确率 = TP/(TP+FP)，这里防止分母为0，赋予最小值1
        if (self.__TP + self.__FP) == 0:
            self.__precision = self.__TP / 1
        else:
            self.__precision = self.__TP/(self.__TP + self.__FP)
        # 真负率/负例的覆盖率/特指度 = TN/N
        self.__specificity = self.__TN/self.__N
        # 负例的命中率 = TN/(FN+TN)，这里防止分母为0，赋予最小值1
        if (self.__FN+self.__TN) == 0:
            self.__npv = self.__TN / 1
        else:
            self.__npv = self.__TN/(self.__FN+self.__TN)
        # 假负率 = FN/P
        self.__fnr = self.__FN/self.__P
        # 假正率 = FP/N = 1 - specificity
        self.__fpr = self.__FP/self.__N
        # 准确性 = (TP+TN)/(P+N)
        __accuracy = (self.__TP+self.__TN)/(self.__P+self.__N)
        # 误分率= (FP+FN)/(P+N) = 1 – accuracy
        __errorRate = (self.__FP+self.__FN)/(self.__P+self.__N)
        # F-measure = 2/(1/Precision + 1/recall)，这里防止分母为0，赋予最小值1
        if self.__precision == 0 or self.__recall == 0:
            __f1 = 0.5
        else:
            __f1 = 2/(1/self.__precision + 1/self.__recall)


    """    指定计算临界比例时的各项指标    """
    def calculateindicatorsbypercentage(self, percentage):
        if percentage < 0 or percentage > 1:
            raise TypeError("概率值必须输入数字且应介于0和1之间")
        __percentage = percentage

        data = self.__data.tolist()
        # 将原始数据按照概率值从高到低排序
        data.sort(reverse=True)
        y_true = numpy.array(data)[:, 1]

        # 取前percent数据的序号
        index_greater_than_percent = numpy.arange(round((self.__N+self.__P)*percentage),
                                                  dtype="int")
        # 取前percent数据之后的序号
        index_less_than_percent = numpy.arange(round((self.__N+self.__P)*percentage),
                                               (self.__N+self.__P), 1, dtype="int")
        # 取正例的序号
        index_positive = numpy.where(y_true == 1)[0]
        # 取负例的序号
        index_negative = numpy.where(y_true == 0)[0]

        # 计算各指标值
        self.__TP = len(set(index_greater_than_percent).intersection(set(index_positive)))
        self.__FP = len(set(index_greater_than_percent).intersection(set(index_negative)))
        self.__FN = len(set(index_less_than_percent).intersection(set(index_positive)))
        self.__TN = len(set(index_less_than_percent).intersection(set(index_negative)))
        # 真正率/灵敏度/覆盖率/召回率/查全率 = TP/P
        self.__recall = self.__TP/self.__P
        # 命中率/查准率/精确率 = TP/(TP+FP)，这里防止分母为0，赋予最小值1
        if (self.__TP + self.__FP) == 0:
            self.__precision = self.__TP / 1
        else:
            self.__precision = self.__TP/(self.__TP + self.__FP)
        # 真负率/负例的覆盖率/特指度 = TN/N
        self.__specificity = self.__TN/self.__N
        # 负例的命中率 = TN/(FN+TN)，这里防止分母为0，赋予最小值1
        if (self.__FN + self.__TN) == 0:
            self.__npv = self.__TN / 1
        else:
            self.__npv = self.__TN/(self.__FN+self.__TN)
        # 假负率 = FN/P
        self.__fnr = self.__FN/self.__P
        # 假正率 = FP/N = 1 - specificity
        self.__fpr = self.__FP/self.__N
        # 准确性 = (TP+TN)/(P+N)
        self.__accuracy = (self.__TP + self.__TN)/(self.__P + self.__N)
        # 误分率= (FP+FN)/(P+N) = 1 – accuracy
        self.__errorRate = (self.__FP+self.__FN)/(self.__P+self.__N)
        # F-measure = 2/(1/Precision + 1/recall)，这里防止分母为0，赋予最小值1
        if self.__precision == 0 or self.__recall == 0:
            self.__f1 = 0.5
        else:
            self.__f1 = 2/(1/self.__precision + 1/self.__recall)

    """    按100份计算各分位点的指标，画出Lift图
           参数:
               divisiontype：划分类型，允许两种：percentage按数量等份100份，probability按概率等分100份
           返回：
               axis_x：图形横坐标值，即各分位点
               axis_y：图形纵坐标值，即各分位点的Lift值
               max(axis_y)：最大Lift值 
    """
    def liftgraph(self, divisiontype='percentage'):
        i = 0
        axis_x = numpy.arange(0, 1.01, 0.01, dtype="float")
        axis_y = numpy.arange(0, 1.01, 0.01, dtype="float")
        if divisiontype == 'probability':
            while i < 100:
                self.calculateindicatorsbyprobability(1-axis_x[i])
                if (self.__TP + self.__FP) == 0:
                    axis_y[i] = (self.__TP) / (self.__P / (self.__P + self.__N))
                else:
                    axis_y[i] = (self.__TP/(self.__TP + self.__FP))/(self.__P/(self.__P + self.__N))
                i += 1
        elif divisiontype == 'percentage':
            while i < 100:
                self.calculateindicatorsbypercentage(axis_x[i])
                if (self.__TP + self.__FP) == 0:
                    axis_y[i] = (self.__TP) / (self.__P / (self.__P + self.__N))
                else:
                    axis_y[i] = (self.__TP/(self.__TP + self.__FP))/(self.__P/(self.__P + self.__N))
                i += 1

        plt.plot(axis_x, axis_y, 'b', label='Max Lift: %0.2f' % max(axis_y))
        plt.legend(loc='lower right')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, axis_y.max()])
        plt.ylabel('lift')
        plt.xlabel(divisiontype)
        plt.show()
        return axis_x, axis_y, max(axis_y)

    """    按100份计算各切分的指标，画出ROC图和洛伦兹曲线图
           参数：
               divisiontype：划分类型，允许两种：percentage按数量等份100份，probability按概率等分100份
           返回：
               axis_x：图形横坐标值，即各分位点
               axis_y_recall：图形纵坐标值，即各分位点的召回率
               axis_y_precision：图形纵坐标值，即各分位点的查准率
               auc：ROC曲线下面积
    """
    def roclorenzgraph(self, divisiontype='percentage'):
        i = 0
        axis_x = numpy.arange(0, 1.01, 0.01, dtype="float")
        axis_y_recall = numpy.arange(0, 1.01, 0.01, dtype="float")
        #axis_y_precision = numpy.arange(0, 1.01, 0.01, dtype="float")
        axis_y_precision = numpy.zeros(101, dtype="float")
        if divisiontype == 'probability':
            while i < 100:
                self.calculateindicatorsbyprobability(1-axis_x[i])
                axis_y_recall[i] = self.__recall
                axis_y_precision[i] = self.__precision
                i += 1
        elif divisiontype == 'percentage':
            while i < 100:
                self.calculateindicatorsbypercentage(axis_x[i])
                axis_y_recall[i] = self.__recall
                axis_y_precision[i] = self.__precision
                i += 1

        auc = numpy.trapz(axis_y_recall, axis_x)
        plt.plot(axis_x, axis_y_recall, 'b', label='AUC: %0.2f' % auc)
        plt.plot(axis_x, axis_y_precision, 'g', label='Max precision: %0.2f' % max(axis_y_precision))
        plt.legend(loc='lower right')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.plot([0, 1], [0, 1], 'r--')
        plt.xlabel(divisiontype)
        plt.show()
        return axis_x, axis_y_recall, axis_y_precision, auc

    """    按100份计算各切分的指标，画出KS图
           参数：
               divisiontype：划分类型，允许两种：percentage按数量等份100份，probability按概率等分100份
           返回：
               axis_x：图形横坐标值，即各分位点
               axis_y_recall：图形纵坐标值，即各分位点的召回率
               axis_y_precision：图形纵坐标值，即各分位点的查准率
               auc：ROC曲线下面积
               max(axis_y_ks)：最大KS值
    """
    def ksgraph(self,divisiontype='percentage'):
        i = 0
        axis_x = numpy.arange(0, 1.01, 0.01, dtype="float")
        axis_y_recall = numpy.arange(0, 1.01, 0.01, dtype="float")
        axis_y_fpr = numpy.arange(0, 1.01, 0.01, dtype="float")
        # axis_y_ks = numpy.arange(0, 1.01, 0.01, dtype="float")
        axis_y_ks = numpy.zeros(101, dtype="float")
        if divisiontype == 'probability':
            while i < 100:
                self.calculateindicatorsbyprobability(1-axis_x[i])
                axis_y_recall[i] = self.__recall
                axis_y_fpr[i] = self.__fpr
                axis_y_ks[i] = axis_y_recall[i] - axis_y_fpr[i]
                i += 1
        elif divisiontype == 'percentage':
            while i < 100:
                self.calculateindicatorsbypercentage(axis_x[i])
                axis_y_recall[i] = self.__recall
                axis_y_fpr[i] = self.__fpr
                axis_y_ks[i] = axis_y_recall[i] - axis_y_fpr[i]
                i += 1

        auc = numpy.trapz(axis_y_recall, axis_x)
        plt.plot(axis_x, axis_y_recall, 'r', label='AUC: %0.2f' % auc)
        plt.plot(axis_x, axis_y_fpr, 'g', )
        plt.plot(axis_x, axis_y_ks, 'b', label='Max KS: %0.2f' % max(axis_y_ks))
        plt.legend(loc='lower right')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel(divisiontype)
        plt.show()
        return axis_x, axis_y_recall, axis_y_ks, auc, max(axis_y_ks)

    """
        在同一个板上画所有的图
    """
    def allgraph(self, divisiontype='percentage'):
        print()

    """
        按100份计算各切分的指标，输出100份的各项指标
        参数：
            divisiontype：划分类型，允许两种：percentage按数量等份100份，probability按概率等分100份
        返回：101*6的二维numpy数组，6列含义分别为：分位点、召回率、查准率、特指度、准确性、F-measure
    """
    def getindicators(self, divisiontype='percentage'):
        i = 0
        threshold = numpy.arange(0, 1.01, 0.01, dtype="float")
        indicators = numpy.arange(0, 1.01, 0.01, dtype="float")
        recall = numpy.zeros(101, dtype="float")
        precision = numpy.zeros(101, dtype="float")
        specificity = numpy.zeros(101, dtype="float")
        accuracy = numpy.zeros(101, dtype="float")
        f = numpy.zeros(101, dtype="float")
        if divisiontype == 'probability':
            while i < 100:
                self.calculateindicatorsbyprobability(1-threshold[i])
                recall[i] = self.__recall
                precision[i] = self.__precision
                specificity[i] = self.__specificity
                accuracy[i] = self.__accuracy
                f[i] = self.__f1
                i += 1
        elif divisiontype == 'percentage':
            while i < 100:
                self.calculateindicatorsbypercentage(threshold[i])
                recall[i] = self.__recall
                precision[i] = self.__precision
                specificity[i] = self.__specificity
                accuracy[i] = self.__accuracy
                f[i] = self.__f1
                i += 1
        indicators = numpy.c_[indicators, recall]
        indicators = numpy.c_[indicators, precision]
        indicators = numpy.c_[indicators, specificity]
        indicators = numpy.c_[indicators, accuracy]
        indicators = numpy.c_[indicators, f]
        return indicators

"""    当直接运行本类时，打印默认初始化值，不做任何计算    """
if __name__ == "__main__":
    cmbme = ConfusionMatrixModelEvaluation()
    print(cmbme)
