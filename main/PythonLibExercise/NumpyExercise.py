"""
    本文通过案例介绍numpy常用语法，作为后续参考
"""

import numpy

# 创建2*3*5维数组列表
a = [[[8, 2, 3, 4, 5], [6, 7, 8, 9, 0], [1, 3, 5, 7, 9]],
     [[1, 2, 3, 4, 5], [6, 7, 8, 9, 0], [1, 3, 5, 7, 9]]]
# 将列表转换为numpy数组
b = numpy.array(a)

"""
    快速初始化数组
"""
print(numpy.full((20, 2), 0.99))         # 创建一个 20行*2列矩阵，初始化全部填为0.99
print(numpy.zeros(10, dtype=int))        # 创建一个 1行*10列数组，初始化全部填为0
print(numpy.ones((3, 5), dtype=int))     # 创建一个 3行*5列数组，初始化全部填为1
print(numpy.arange(start=3, stop=9, step=2, dtype='int'))     # 创建一个等差数组，起始3终止9步长2
print(numpy.linspace(3, 10, 7))             # 创建等差数列：包含7个值，这7个数均匀的分配在3～10区间内
print(numpy.logspace(0, 15, 11, base=2))    # 创建等比数列：创建11个值，从2的-5次方到2的15次方
print(numpy.random.rand(3, 3))              # 创建一个3x3的，在0～1区间的随机数组成的数组
print(numpy.random.normal(0, 1, (5, 5)))    # 创建一个5x5的，均值为0，方差为1，正态分布的随机数数组
print(numpy.eye(3))       # 创建一个3x3的单位矩阵，就是对角线为1，其他均为0
print(numpy.empty(3))     # 创建一个由3个整形组成的未初始化的数组,数组的值是内存空间中的任意值
print(numpy.random.random_integers(3, 9, size=5))       # 创建5个随机整数，位于闭区间 [3, 9]

"""
    查看数组
"""
print(b.size)         # 查看数组中元素的总个数
print(b.shape)        # 查看数组形状
print(b.dtype)        # 查看数组元素的类型
print(b.ndim)         # 查看数组的维度
print(b.itemsize)     # 查看数组中每个元素的字节数
print(b.T)            # 查看数组的转置
print(list(b.flat))   # 返回一个数组的迭代器，就是把所有元素展平成一维，flat不能直接打印，需转成list
print(b.nbytes)       # '查看数组中所有元素在内存所占总字节数

# slice切片
a = numpy.random.randint(0, 100, size=20)
print('原始数组：', a)
s = slice(5, 14, 2)      # slice是定义一个索引，始于5终于14步长2，结果是[5,7,9,11,13]
print('：', a[s])        # 切片数组a[s]的意思就是取第5,7,9,11,13个元素值
print('直接切片数组：', a[2:9:2])    # 就相当于先slice(2, 9, 2)
print(a[5])    # 对单个元素切片
print(a[16:])  # 对始于索引的元素进行切片
print(a[2:5])  # 对索引之间的元素进行切片

a = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(a)
print(a[..., 1])    # 查看第二列的元素
print(a[1, ...])    # 查看第二行的元素
print(a[..., 1:])   # 从第二列向后切片所有元素

"""
    条件查询
"""
a = numpy.arange(start=3, stop=19, step=2, dtype='int')
b = numpy.where((a > 5) & (a < 13))    # b返回的是符合条件的索引值
print(a)
print(b)
print(a[b])

"""
    关于排序
"""
# 普通list排序，是元素跟着一起排，类似于数据库表的排序操作
t1 = t2 = [[0, 1], [5, 0], [3, 8], [4, 3]]
t1.sort(reverse=True)        # 默认按照第一列的值排序，降序
print(t1)
t2.sort(key=lambda x: x[1])  # 指定按照第二列的值排序
print(t2)
# 但是numpy的sort排序，完全是矩阵排序，每个元素不跟着走的，完全只看横竖
t1 = [[0, 1], [5, 0], [3, 8], [4, 3]]
print(numpy.sort(t1, axis=0))    # 竖着排
print(numpy.sort(t1, axis=1))    # 横着排
t2 = numpy.array(t1)
print(t2[:, 1])
# numpy也可以按元素跟着走的排序，用lexsort函数
# https://www.cnblogs.com/liyuxia713/p/7082091.html
t1 = numpy.array([[0, 1], [5, 0], [3, 8], [4, 3]])
print("lexsort", t1[numpy.lexsort(t1.T)])


"""
    矩阵内部的计算、变形
"""
a = numpy.random.normal(0, 1, (1, 10))
# print(a.reshape(2, 5))      # 把一个一维矩阵改成2*5的二维
b = a.reshape(2, 5)
print(numpy.abs(b))           # 求矩阵每个元素的绝对值，类似函数还有square、exp、log、
print(numpy.sign(b))          # ceil、floor、sign、modf、三角函数、反三角函数
print(numpy.sum(b, axis=0))   # 按列求和，类似函数还有mean、std、var、min、max、argmin、argmax……
print(numpy.unique(b))
a = numpy.arange(9)
b = numpy.split(ary=a, indices_or_sections=3, axis=0)         # 把一个数组切割成多个数组，必须能够等分，否则会报错
c = numpy.array_split(ary=a, indices_or_sections=2, axis=0)   # 把一个数组切割成多个数组，可以不等分
print(b)


"""
    两个矩阵的运算，基本上要求形状相同
"""
a = numpy.arange(start=0, stop=10, step=1)
a = a.reshape(2, 5)
b = numpy.arange(start=6, stop=16, step=1)
b = b.reshape(2, 5)
print(numpy.add(a, b))          # 两矩阵每个元素相加，类似函数还有subtract、multiply、divide、power
print(numpy.intersect1d(a, b))  # 计算两个矩阵的公共元素，并返回有序结果
print(numpy.union1d(a, b))      # 计算两个矩阵的并集，去重并返回有序结果
print(numpy.dot(a, b.T))        # 两个矩阵相乘

"""
    两个矩阵的堆叠组合
"""
print(numpy.c_[a, b])    # 按列连接两个矩阵，就是把两矩阵上下相加，要求列数相等
print(numpy.r_[a, b])    # 按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等
# 似乎hstack() vstack() stack() dstack() column_stack() row_stack()都是堆叠函数

"""
    其他神奇的函数
"""
# map()会根据提供的函数对指定序列做映射，map(function, iterable, ...)
# 其中function是指定函数（可自定义），iterable是数据，map函数本身无法打印，需转成list
print(list(map(numpy.square, a)))      # 返回a的每个元素的平方
print(list(map(lambda x: x + 1, a)))   # 返回a的每个元素的自定义函数
