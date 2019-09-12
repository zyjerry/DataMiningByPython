"""
    0.前言：
    本程序用tensorflow完成一个一元线性回归的模型拟合，并画图展示。
    传统的线性回归模型，使用最小二乘法计算直接得到回归系数，但是tensorflow常用梯度下降算法，可达到同样效果。
"""

import numpy
import tensorflow.python as tf
import matplotlib.pyplot as plt


class TensorFlowLinearRegression0:
    """    私有属性声明    """
    __x_data = numpy.float32(numpy.random.rand(50)*5)
    __e_data = numpy.float32(numpy.random.normal(0, 1, 50) * 2)
    __w_data = 2.7
    __b_data = -5.3
    __y_data = __w_data * __x_data + __b_data + __e_data

    """
        构造函数，传入指定参数：斜率和截距
    """
    def __init__(self, w=2.7, b=-5.3):
        self.__w_data = w
        self.__b_data = b
        self.__y_data = self.__w_data * self.__x_data + self.__b_data + self.__e_data

    """    
        当print该类对象时返回的内容    
    """
    def __str__(self):
        printstr = "TensorFlowLinearRegression0类：\n初始数据规模："
        printstr += str(self.__x_data.size) + "，w=" + str(self.__w_data) \
                    + ", b=" + str(self.__b_data)
        return printstr

    """
        训练数据
    """
    def train(self):
        """
            1、构造tensorflow的基本算子、算法。注意这一步都是在“定义”和“构造”，不是真正的模型训练和计算
        """
        # 先构造一个线性模型 y = w*x + b，这里w被赋予随机值，介于-1和1之间，b分配一个变量并赋值为0
        w = tf.Variable(tf.random_uniform([1], -1.0, 1.0))
        b = tf.Variable(tf.zeros([1]))
        y = tf.mul(w, self.__x_data) + b
        # 定义损失函数（方差）和优化算法（梯度下降），目标就是最小化损失函数
        loss = tf.reduce_mean(tf.square(y - self.__y_data))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05)
        train = optimizer.minimize(loss)
        # 初始化变量
        init = tf.global_variables_initializer()

        """
            2、正式训练
        """
        # 建立会话
        sess = tf.Session()
        # 这个时候才开始真正地计算
        sess.run(init)
        print('初始化参数：w=', sess.run(w), ', b=', sess.run(b))
        # 拟合平面，过程就是执行100遍梯度下降算法，得到最佳的w和b
        for step in numpy.arange(0, 101):
            sess.run(train)
            if step % 10 == 0:
                print(step, sess.run(w), sess.run(b))

        """
            3、画图
        """
        plt.scatter(self.__x_data, self.__y_data, marker='.', color='red', s=40, label='First')
        plt.plot([numpy.min(self.__x_data), numpy.max(self.__x_data)],
                 [sess.run(w)*numpy.min(self.__x_data)+sess.run(b),
                  sess.run(w)*numpy.max(self.__x_data)+sess.run(b)],
                 'b')
        plt.show()

        """
            4、任务完成, 关闭会话.
        """
        sess.close()


if __name__ == "__main__":
    tflr0 = TensorFlowLinearRegression0(1.5, -1)
    print(tflr0)
    tflr0.train()
