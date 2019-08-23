"""
    关于numpy库的典型使用方法
"""

import numpy

# 创建2*3*5维数组列表
a = [ [ [8, 2, 3, 4, 5],[6, 7, 8, 9, 0],[1, 3, 5, 7, 9] ],
      [ [1, 2, 3, 4, 5],[6, 7, 8, 9, 0],[1, 3, 5, 7, 9] ]
    ]
# 将列表转换为numpy数组
b = numpy.array(a)
print('查看数组：', b)
print('查看数组中元素的个数：', b.size)
print('查看数组形状：', b.shape)
print('查看数组元素的类型：', b.dtype)
print('查看数组的维度：', b.ndim)
print('查看数组中元素的个数(从0开始)：', b.itemsize)
print('查看数组的转置：', b.T)
print('返回一个数组的迭代器，：', b.flat)        # 对flat赋值将导致整个数组的元素被覆盖
print('查看数组中的元素在内存所占字节数：', b.nbytes)

# slice切片
a = numpy.arange(10)
print('原始数组：', a)
s = slice(2, 7, 2)
print('切片数组：', a[s])
print('直接切片数组：', a[2:9:2])
print('对单个元素切片：', a[5])
print('对始于索引的元素进行切片：', a[2:])
print('对索引之间的元素进行切片：', a[2:5])


a = numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print('3*3的数组是：', a)
print('第二列的元素是：', a[..., 1])
print('第二行的元素是：', a[1, ...])
print('从第二列向后切片所有元素：', a[..., 1:])

c_range = numpy.logspace(-5, 15, 11, base=2)
gamma_range = numpy.logspace(-9, 3, 13, base=2)
print(c_range)
print(gamma_range)
