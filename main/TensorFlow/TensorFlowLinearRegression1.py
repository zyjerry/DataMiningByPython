"""
    0.前言：
    本程序用tensorflow完成一个二元一次线性回归的模型拟合。
    与TensorFlowLinearRegression0不同的是，本程序引入了tf.Graph()，并采用with的写法，功能差不多。
    Author：Jerry Zhang
    Email：zyjerry@gmail.com
"""

import numpy
import tensorflow.python as tf


class TensorFlowLinearRegression1:
    """    私有属性声明    """
    # 构造测试数据，x是一个2*100数组，赋予0-5之间的随机数
    __x_data = numpy.float32(numpy.random.rand(2, 100) * 5)
    # 构造e，就是一个随机的符合正态分布的扰动值
    __e_data = numpy.float32(numpy.random.normal(0, 1, (2, 100))) * 2
    # 构造y = w*x + b + e，即最终是一个包含100个元素的数组，这里w包含2个系数，对应x_data里的2行
    __w_data = [2.7, -1.8]
    __b_data = 5.3
    __y_data = numpy.dot(__w_data, __x_data) + __b_data + __e_data

    """
        构造函数，传入指定参数：斜率和截距
    """
    def __init__(self, weight=[2.7, -1.8], bias=5.3):
        self.__w_data = weight
        self.__b_data = bias
        self.__y_data = numpy.dot(self.__w_data, self.__x_data) + self.__b_data + self.__e_data

    """    
        当print该类对象时返回的内容    
    """
    def __str__(self):
        printstr = "TensorFlowLinearRegression0类：\n初始数据规模："
        printstr += str(self.__x_data.shape) + "，w=" + str(self.__w_data) \
            + "，b=" + str(self.__b_data)
        return printstr

    """
        训练数据
    """
    def train(self):
        """
            1、构造tensorflow的基本算子、算法。注意这一步都是在“定义”和“构造”，不是真正的模型训练和计算
            这里需特别注意一下训练数据和学习率的关系：
            本案例中，训练数据X都在0-1之间，学习率取0.5是比较恰当的。
            但是，当训练数据越大的时候，学习率越要变小，例如X在0-5之间的话，学习率取0.05较合适。
            个人感觉：训练数据取值越大，如果学习率不降的话，在每一步梯度计算时，容易“步子太大扯着蛋”，
                     即所谓的“梯度爆炸”，最终无法收敛导致系数越来越大直到溢出，算不出来了。
        """
        # 先构造一个数据流图，定义线性模型 y = w*x + b，这里w被赋予随机值，介于-1和1之间，b分配一个变量并赋值为0
        temp_graph = tf.Graph()
        with temp_graph.as_default():
            tf_v_w = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
            tf_v_b = tf.Variable(tf.zeros([1]))
            tf_v_y = tf.matmul(tf_v_w, self.__x_data) + tf_v_b
            # 定义损失函数（方差）和优化算法（梯度下降），目标就是最小化损失函数
            loss = tf.reduce_mean(tf.square(tf_v_y - self.__y_data))
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
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
            print('初始化参数：w=', sess.run(tf_v_w), ', b=', sess.run(tf_v_b))
            # 拟合平面，过程就是执行100遍梯度下降算法，得到最佳的w和b
            for step in numpy.arange(0, 101):
                sess.run(train)
                if step % 10 == 0:
                    print("第%u步：权重：%s，偏置：%f，损失：%f" %
                          (step, tf_v_w.eval(), tf_v_b.eval(), loss.eval()))
                # 将训练完毕的参数保存
                self.__w_data = tf_v_w.eval()
                self.__b_data = tf_v_b.eval()

    """
        获取训练好的参数
    """
    def getparameters(self):
        return self.__w_data, self.__b_data

    """
        预测数据。
        参数：
            x：一个2*任意列的数组
        输出：
            pre：一个与x列数一致的一维数组
    """
    def predict(self, x_pre=None):
        pre = numpy.dot(self.__w_data, x_pre) + self.__b_data
        return pre


if __name__ == "__main__":
    tflr1 = TensorFlowLinearRegression1(weight=[1.5, 2.9], bias=-1)
    print(tflr1)
    tflr1.train()
    w, b = tflr1.getparameters()
    print("trained w is: ", w)
    print("trained b is: ", b)
    x = numpy.random.rand(2, 5)
    print("test x is:", x)
    y = tflr1.predict(x)
    print("predict y is: ", y)
