import pandas
import sas7bdat

# 将数组转换成Series，分别指定值和索引，如果不指定索引，则默认0、1、2……，注意这里的索引值不一定是同类型的
countries = ['USA', 'UK', 'RPC', 'JP'];
my_index = ['a', 200, 300, 400]
sp = pandas.Series(countries, my_index)
print(sp)
print(sp['a'])

# 将字典数据结构转换成Series
my_dict = {'a': 'USA', 'b': 'UK', 'c': 'RPC', 'd': 'JP'}
md = pandas.Series(my_dict)
print(md)

# Series之间的计算
s1 = pandas.Series([1, 3, 5, 7, 9], ['USA', 'UK', 'RPC', 'JP', 'KR'])
s2 = pandas.Series([2, 3, 5, 1, 8], ['USA', 'UK', 'RPC', 'AUSTR', 'RU'])
print(s1+s2)
print(s1-s2)
print(s1*s2)
print(s1/s2)

# DataFrame单表操作：增、删、改、查
df = {
    'NAME': pandas.Series(['John', 'Kitty', 'Danny', 'Alice'], index=[0, 1, 2, 3]),
    'AGE':  pandas.Series([8, 10, 7, 9], index=[0, 1, 2, 4]),
    'NATIONALITY': pandas.Series(['USA', 'UK', 'RPC', 'AUSTR', 'RU'], index=[0, 1, 4, 5, 8])
}
my_df = pandas.DataFrame(df)
print(my_df)
print(my_df.loc[1])     # 显示某一行
my_df.dropna(axis=1)    # 删除存在空值的行
my_df.dropna(axis=0)    # 删除存在空值的列
print(my_df)
my_df.fillna(value=99)  # 填充空值
print(my_df)
print(my_df.shape)

# DataFrame多表操作：关联查询、追加记录、横向合并

# 数据输入
# 和SAS数据文件的交互，pandas.read_sas处理gb2312的中文是乱码，这里使用sas7bdat包中的函数，先转换成DataFrame
#sasdata = pandas.read_sas('D:\\06-JerryTech\\pf_data1.sas7bdat')
sasdata = sas7bdat.SAS7BDAT('D:\\06-JerryTech\\pf_data1.sas7bdat', encoding="gb2312").to_data_frame()
writer = pandas.ExcelWriter('D:\\06-JerryTech\\pf_data11.xlsx')
sasdata.to_excel(writer)
writer.save()

# 数据输出
