"""
    0.前言：
    本程序用tensorflow完成一个逻辑回归的模型拟合。
    已经有一份处理好且自变量均有效的宽表数据dd_df.xlsx（见本目录下SAS数据文件）：
    loan_id为流水号，y是因变量（二分类，其中y=1表示坏客户），其余均为自变量。
    通过这个案例可以更好地理解逻辑回归的本质原理思想、模型函数、损失函数、梯度下降的概念
    Author：Jerry Zhang
    Email：zyjerry@gmail.com
"""
import numpy
import pandas
import tensorflow.python as tf
import sklearn.linear_model


class TensorFlowLogicRegression:
    """
        私有属性声明和初始化
    """
    __x_train = None
    __y_train = None
    __weight = None
    __bias = 1.0

    """
        构造函数，初始化数据
        参数：
            x_train：二维numpy数组
            y_train：一维numpy数组
    """
    def __init__(self, x_train=None, y_train=None):
        print("TensorFlowLogicRegression init........")
        if x_train.ndim != 2:
            raise TypeError("输入的X训练数据不是二维矩阵")
        if y_train.ndim != 1:
            raise TypeError("输入的Y训练数据不是一维矩阵")
        self.__x_train = x_train
        self.__y_train = y_train
        self.__weight = numpy.random.rand(1, self.__x_train.shape[1])

        print("x_train.shape:", x_train.shape)
        print("y_train.shape:", y_train.shape)

    """
        当print该类对象时返回的内容
    """
    def __str__(self):
        printstr = "TensorFlowLogicRegression类：\n训练数据规模：X="
        printstr += str(self.__x_train.shape) + "，Y=" + str(self.__y_train.shape)
        return printstr

    """
        训练数据
    """
    def train(self):
        """
            1、构造tensorflow的基本算子、算法。注意这一步都是在“定义”和“构造”，不是真正的模型训练和计算
        """
        # 先构造一个数据流图
        temp_graph = tf.Graph()
        with temp_graph.as_default():
            # 定义占位符，表示待训练的数据集，用这种方式最后运行train的时候总是报错，暂无法解决：
            # You must feed a value for placeholder tensor 'x' with dtype float and shape [?,?]
            # x = tf.placeholder(dtype=tf.float32, shape=[None, None], name='x')
            # y = tf.placeholder(dtype=tf.float32, shape=[None], name='y')

            # 定义待训练的参数w和b，weight被赋予随机值，介于-1和1之间，bias分配一个变量并赋值为0
            weight = tf.Variable(tf.random_uniform([1, self.__x_train.shape[1]], -1.0, 1.0))
            bias = tf.Variable(tf.zeros([1]))

            # 定义二分类的sigmoid模型 y = 1/(1+exp-(w*x + b))
            # y_pre = tf.div(1.0,
            #                tf.add(1.0,
            #                       tf.exp(tf.neg(tf.reduce_sum(tf.multiply(weight, self.__x_train),
            #                                                   1
            #                                                  ) + bias)
            #                             )
            #                      )
            #               )
            # 也可以直接利用tf的sigmoid函数
            y_pre = tf.sigmoid(tf.reduce_sum(tf.multiply(weight, self.__x_train), 1) + bias)

            # 定义损失函数为对数损失函数(-y*log(y_pre) - (1-y)*log(1-y_pre))/样本数
            loss0 = self.__y_train * tf.log(y_pre)
            loss1 = (1 - self.__y_train) * tf.log(1 - y_pre)
            loss = tf.reduce_sum(- loss0 - loss1) / self.__x_train.shape[0]
            # 定义优化算法（梯度下降），目标就是最小化损失函数
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
            train = optimizer.minimize(loss)
            # 初始化变量
            init = tf.global_variables_initializer()

        """
            2.正式训练
        """
        # 建立会话
        with tf.Session(graph=temp_graph) as sess:
            # 这个时候才开始真正地计算
            sess.run(init)
            print('初始化参数：weight=', sess.run(weight), ', bias=', sess.run(bias))
            # 拟合平面，过程就是执行1000遍梯度下降算法，得到最佳的w和b
            for step in range(1000):
                sess.run(train)
                if step % 100 == 0:
                    print("第%u步：权重：%s，偏置：%f，损失：%f" %
                          (step, weight.eval(), bias.eval(), loss.eval()))
                self.__weight = weight.eval()
                self.__bias = bias.eval()

    """
        获取训练后的参数
    """
    def getparameters(self):
        return self.__weight, self.__bias


if __name__ == "__main__":
    # 读入数据
    data1 = pandas.read_excel('D:\\06-JerryTech\\dd_df.xlsx', sheet_name='Sheet1')
    data1[['y']] = data1[['y']].astype('str')
    useful_columns = pandas.Series(data1.columns)
    useful_columns = useful_columns[useful_columns.values != 'loan_id']
    useful_columns = useful_columns[useful_columns.values != 'y']
    x_train_data = numpy.array(data1[useful_columns]).astype(numpy.float)
    y_train_data = numpy.array(data1['y']).astype(numpy.float)

    # 用tensorflow训练逻辑回归
    tflr = TensorFlowLogicRegression(x_train=x_train_data, y_train=y_train_data)
    print(tflr)
    tflr.train()
    w, b = tflr.getparameters()
    print("tensorflow trained w is: ", w)
    print("tensorflow trained b is: ", b)

    # 用sklearn训练逻辑回归，可以对比结果：weight不太一样，bias基本一致
    # 结论：建模是个无限逼近的“差不多”艺术*^_^*
    lr = sklearn.linear_model.LogisticRegression(penalty='none', C=1, solver='sag')
    lr.fit(x_train_data, y_train_data)
    print('lr coef:', lr.coef_)
    print('lr intercept:', lr.intercept_)
