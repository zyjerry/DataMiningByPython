"""
    0.前言：
    本程序用tensorflow完成一个基本的三层神经网络模型：
    第一层：输入层，原始数据；
    第二层：隐藏层，含有若干个神经元节点；
    第三层：输出层，结果，基于本案例的二分类属性，输出层即为一个神经元。
    已经有一份处理好且自变量均有效的宽表数据dd_df.xlsx（见本目录下SAS数据文件）：
    loan_id为流水号，y是因变量（二分类，其中y=1表示坏客户），其余均为自变量。
    Author：Jerry Zhang
    Email：zyjerry@gmail.com
"""

import numpy
import pandas
import tensorflow.python as tf


class TensorFlowMLP:
    """
        私有属性声明和初始化
    """
    __x_train = None
    __y_train = None
    __weights = None
    __biases = None

    """
        构造函数，初始化数据
        参数：
            x_train：二维numpy数组
            y_train：一维numpy数组
    """
    def __init__(self, x_train=None, y_train=None):
        self.__x_train = x_train
        self.__y_train = y_train
        self.__y_train.shape = (y_train.shape[0], 1)
        print("X/Y shape is : ", self.__x_train.shape, self.__y_train.shape)
        print("X/Y dtype is : ", self.__x_train.dtype, self.__y_train.dtype)

    """
        训练数据
    """
    def train(self):
        """
            1、构造tensorflow的基本结构。注意这一步都是在“定义”和“构造”，不是真正的模型训练和计算
        """
        # 先构造一个数据流图
        temp_graph = tf.Graph()
        with temp_graph.as_default():
            # 构造输入数据占位符
            # tf_x = tf.placeholder(dtype=tf.float32, shape=(None, 10), name='features')
            # tf_y = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='targets')

            # 构造两层的权重和系数，隐藏层定义为8个神经元（为什么是8个？没啥道理，随便试的），输出层为1个神经元
            weights = {
                       # 原始变量为10个，所以隐藏层总参数个数是10*8
                       'h1': tf.Variable(tf.truncated_normal([10, 8], stddev=0.1)),
                       # 隐藏层定义为8个神经元，输出层为1个神经元，所以输出层总参数个数是8*1
                       'out': tf.Variable(tf.truncated_normal([8, 1], stddev=0.1))
                      }
            biases = {
                       'b1': tf.Variable(tf.zeros([8])),
                       'out': tf.Variable(tf.zeros([1]))
                     }

            # 定义2层感知机，一层隐藏层，一层输出层
            # 隐藏层是一个线性映射 + reLu激活函数（为什么是reLu？可以避免梯度太快消失）
            layer_hidden = tf.add(tf.matmul(self.__x_train, weights['h1']), biases['b1'])
            layer_hidden = tf.nn.relu(layer_hidden)
            # 输出层是一个线性映射 + sigmoid激活函数，以便更好拟合目标的0/1值
            layer_out = tf.matmul(layer_hidden, weights['out']) + biases['out']
            layer_out = tf.sigmoid(layer_out)

            # 定义损失函数为均方差（这里没有用对数似然函数，看看结果跟逻辑回归是否差不多）
            loss = tf.reduce_mean(tf.reduce_sum(tf.square(self.__y_train - layer_out)))
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
            train = optimizer.minimize(loss, name='train')

            # 预测
            # correct_prediction = tf.equal(tf.argmax(tf_y, 1), tf.argmax(layer_out, 1))
            # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        """
            2、训练
        """
        with tf.Session(graph=temp_graph) as sess:
            sess.run(tf.global_variables_initializer())
            print('初始化参数：weight=', sess.run(weights), ', bias=', sess.run(biases))

            # 循环10次迭代训练
            for epoch in range(10):
                # sess.run(train, feed_dict={tf_x: self.__x_train, tf_y: self.__y_train})
                sess.run(train)
                print("Epoch: %03d | AvgCost: %.3f" % (epoch + 1, loss.eval()), end="\n")
            print("weights[h1]:", weights['h1'].eval())
            print("weights[out]:", weights['out'].eval())
            print("biases[h1]:", biases['b1'].eval())
            print("biases[out]:", biases['out'].eval())

            self.__weights = weights.eval()
            self.__biases = biases.eval()

    """
        预测数据
    """
    def predict(self, x_test):


if __name__ == "__main__":
    # 读入数据
    data1 = pandas.read_excel('D:\\06-JerryTech\\dd_df.xlsx', sheet_name='Sheet1')
    data1[['y']] = data1[['y']].astype('str')
    useful_columns = pandas.Series(data1.columns)
    useful_columns = useful_columns[useful_columns.values != 'loan_id']
    useful_columns = useful_columns[useful_columns.values != 'y']
    x_train_data = numpy.array(data1[useful_columns]).astype(numpy.float32)
    y_train_data = numpy.array(data1['y']).astype(numpy.float32)

    # 用tensorflow多层感知机训练
    tfm = TensorFlowMLP(x_train=x_train_data, y_train=y_train_data)
    tfm.train()
