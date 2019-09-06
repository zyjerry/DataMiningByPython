import numpy
import pandas
import sas7bdat

"""
    本文通过案例介绍pandas常用语法，作为后续参考
"""

"""
    Series：是一个高级的list，可以对应表格中的一列，包含index和data两部分数据结构，index用来索引
"""
# 将list转换成Series，分别指定值和索引，如果不指定索引，则默认0、1、2……，注意这里的索引值不必是同类型的
countries = ['USA', 'UK', 'RPC', 'JP']
my_index = ['a', 200, 300, 400]
sp = pandas.Series(countries, my_index)
print(sp)
print(sp['a'])     # 查看索引为a的数据值
print(sp.shape)    # 查看series形状，(4,)，不是一个纯数字

# 将字典数据结构转换成Series，其中key转为index，value转为data
my_dict = {'a': 'USA', 'b': 'UK', 'c': 'RPC', 'd': 'JP'}
md = pandas.Series(my_dict)
print(md)

# Series之间的计算，就是每个元素的计算形成新的Series，用索引外关联，任一Series没有某个索引值的则返回NaN
s1 = pandas.Series([1, 3, 5, 7, 9], ['USA', 'UK', 'RPC', 'JP', 'KR'])
s2 = pandas.Series([2, 3, 5, 1, 8], ['USA', 'UK', 'RPC', 'AUSTR', 'RU'])
print(s1 + s2)
print(s1 - s2)
print(s1 * s2)
print(s1 / s2)


"""
    DataFrame：是一个高级的二维数据表，包含index、column、data三部分
"""
"""
    单表操作：增、删、改、查
"""
# DataFrame的构造，由每一列组成，每一列都有column、data、index，最终形成的表由index外关联
df = {
    'NAME': pandas.Series(['John', 'Kitty', 'Danny', 'Alice'], index=[0, 1, 2, 3]),
    'AGE':  pandas.Series([8, 10, 7, 9], index=[0, 1, 2, 4]),
    'NATIONALITY': pandas.Series(['USA', 'UK', 'RPC', 'AUSTR', 'RU'], index=[0, 1, 4, 5, 8])
}
my_df = pandas.DataFrame(df)
print(my_df)

# DataFrame的快速查看和统计
print(my_df.info())      # 查看表的概况，各列数据类型、缺失情况等
print(my_df.describe())  # 统计信息概览，只针对数值型列，包括均值、标准差、最小值、最大值、25%、50%、75%分位数
print(my_df.quantile(0.22))    # 也可以指定任一分位数查询值
print(my_df.head(10))  # 查看前10行
print(my_df.tail(5))   # 查看后5行
print(my_df.shape)     # 显示行列数

# 增加一列
my_df['NEW_COLUMN'] = pandas.Series(data=[1, 2, 3, 4], index=[0, 1, 2, 3])   # 直接增加一列
print(my_df)
# 增加一行
a = {'NAME': 'James', 'AGE': 11, 'NATIONALITY': 'China', 'NEW_COLUMN': 'nc'}
my_df = my_df.append(a, ignore_index=True)
print(my_df)

# 删
my_df.drop(labels='NEW_COLUMN', axis=1, inplace=True)   # 直接删除一列
my_df.drop(labels=0, axis=0)   # 直接删除一行
my_df.dropna(axis=1)           # 删除存在空值的行
my_df.dropna(axis=0)           # 删除存在空值的列
print(my_df)

# 改
my_df.fillna(value=99)  # 填充空值
my_df['AGE'] = pandas.Series(data=[10, 20, 30, 40], index=[0, 1, 2, 3])   # 直接改整列值
my_df['NAME'] = my_df['NAME'].str.replace('i', 'iii')  # 修改字符串值，其他操作等同于python的str语法
my_df['AGE'] = my_df['AGE'] + 5  # 修改数值型值，其他操作等同于python的数值运算
my_df['NEW_COLUMN'] = my_df['NAME'] + ',' + my_df['NATIONALITY']   # 列和列之间的操作运算
print(my_df)
# 转置
s1 = pandas.Series([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
d1 = pandas.DataFrame(s1.describe())
print(d1.T)


# 查
print(my_df[['NAME', 'AGE']])  # 查看多个列值
print(my_df.loc[1])            # 显示某一行
print(my_df.loc[3:])           # 显示多行,index>=3的
print(my_df.loc[3:5])          # 显示多行,index在3和5之间的
# 条件查询，查询语句可以带参数，注意查询语句里的column名字也是大小写敏感的
# 注意NATIONALITY == NATIONALITY是表示非空的判断
prob = 20
my_df1 = my_df.query('NATIONALITY == NATIONALITY and AGE > %f' % prob)
print(my_df1)

"""
    单表高级查询和统计：统计类、矩阵运算类和GROUP BY
"""
print(my_df.count(axis=0))  # 按列统计行数，NaN不算，类似的函数还有：sum,mean,max,abs,mask,median
print(my_df.count(axis=1))  # 按行统计列数，NaN不算
print(my_df.corr())         # 计算矩阵各列的相关系数，只适用于数值型字段，返回相关系数矩阵，这个函数牛

# 这个groupby要好好说道说道
df = pandas.DataFrame([[1, 1, 2], [1, 2, 3], [2, 3, 4]], columns=["A", "B", "C"])
# 根据A列group，这时group出来的结果，是若干分组，类型是DataFrameGroupBy，奇怪的东西
# 还没法直观打印出来，我理解依然是分组后的二维明细数据，返回值分别是(A的不同值以及其他字段明细值)的组合
grouped = df.groupby('A')
# group后跟一个列名，依然是个奇怪的对象SeriesGroupBy，没法直观打印出来，我理解依然是分组后的单列Series明细
groupedB = df.groupby('A')['B']
# 最后要加上一个统计函数，才能打印出来，相当于 select A, sum(B) from df group by A
# 可用的函数还有：count、mean、median、std、min、max、quantile...
g = df.groupby('A')['B'].sum()
# 还有种写法是先选要统计的列，再选要groupby的字段
gg = df['B'].groupby(df['A']).sum()
# 查看分组明细
for name, group in grouped:
    print(name, group)
# groupby默认在axis=0上进行分组的，通过设置也可以在其他任何轴上进行分组，如axis=1，太别扭了我不举例了


"""
    DataFrame多表操作：表关联、横向合并
"""
# 关联查询
df1 = pandas.DataFrame([[1, 2], [3, 4]], columns=['col1', 'col2'], index=[1, 0])
df2 = pandas.DataFrame([[5, 6], [7, 8]], columns=['col3', 'col4'], index=[0, 1])
df1.join(df2, how='inner')  # 内关联（还有outer外连接、left左连接、right右连接），直接通过index关联

# 横向合并
df1 = pandas.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]})
df2 = pandas.DataFrame({'col2': [4, 5, 7], 'col3': [1, 2, 2]})
print(df1.merge(df2))    # 通过index关联并横向扩展，内关联，col2列不一致的值去掉了
print(df1.merge(df2, how='left', left_on='col1', right_on='col3'))  # 指定左关联字段并横向合并

"""
    数据输入、输出
"""
# 和SAS数据文件的交互，pandas.read_sas处理gb2312的中文是乱码，这里使用sas7bdat包中的函数，先转换成DataFrame
sasdata1 = pandas.read_sas('D:\\06-JerryTech\\pf_data1.sas7bdat')
sasdata2 = sas7bdat.SAS7BDAT('D:\\06-JerryTech\\pf_data1.sas7bdat', encoding="gb2312").to_data_frame()
writer = pandas.ExcelWriter('D:\\06-JerryTech\\pf_data11.xlsx')
sasdata2.to_excel(writer)
writer.save()

"""
    其他常用、有用功能
"""
# 透视表，参数：index行，columns列，values要统计的列，aggfunc统计函数，margins加汇总值，normalize='index'算占比
df = pandas.DataFrame([[1, 11, 3], [4, 17, 6], [7, 14, 9], [7, 11, 12], [4, 14, 15], [1, 17, 18]],
                      columns=["A", "B", "C"])
g = pandas.crosstab(index=df.A, columns=df.B, values=df.C, aggfunc=numpy.mean, margins=True,
                    normalize='index')
