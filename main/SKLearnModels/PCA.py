"""
    0.前言：
    本程序详细演示主成分分析方法在数据降维当中的应用。
    已经有一份处理好且自变量均有效的宽表数据dd_df.xlsx：
    loan_id为流水号，y是因变量（其中y=1为正例，表示坏客户），其余均为自变量换算成的woe值。
"""
import numpy
import pandas
import sklearn.model_selection
import sklearn.decomposition

"""
    1.导入数据，划分训练集和测试集数据
    注意：决策树虽然是针对离散型自变量构建，但训练数据仍须转换成数值型。
    这里直接使用了变量的woe值，个人认为也可以根据自变量针对因变量的响应程度转换成其它有序系数也可。
"""
data1 = pandas.read_excel('D:\\06-JerryTech\\dd_df.xlsx', sheet_name='Sheet1')
data1[['y']] = data1[['y']].astype('str')
useful_columns = pandas.Series(data1.columns)
useful_columns = useful_columns[useful_columns.values != 'loan_id']
useful_columns = useful_columns[useful_columns.values != 'y']
X_train, X_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(data1[useful_columns], data1['y'], test_size=0.2,
                                             random_state=0)
print("useful_columns:", useful_columns)
print("X_train shape:", X_train.shape)

"""
    2.做主成分分析
    PCA的参数n_components，是降维后的维度数量，也可以指定为mle自动降维。
    本案例能够自动降维到9，可通过pca.n_components_查看。
    本案例中，自变量有10个，分别做10、8、4、2个主成分分析，查看结果：
    explained_variance_：投影后的各维度的方差分布，值都是一样的，就是维度数量不同
    explained_variance_ratio_：投影后各特征维度的方差比例
    要注意一个问题是：
    降维后并不是说选取原维度中的一部分数据，而是将原所有数据经过计算产生新的数据。
    详细原理可参考PCA的思想：https://www.cnblogs.com/pinard/p/6239403.html
"""
pca = sklearn.decomposition.PCA(n_components=10)
pca.fit(X_train)
print("explained_variance_10:", pca.explained_variance_)
print("explained_variance_ratio_10:", pca.explained_variance_ratio_)

pca = sklearn.decomposition.PCA(n_components=8)
pca.fit(X_train)
print("explained_variance_8:", pca.explained_variance_)
print("explained_variance_ratio_8:", pca.explained_variance_ratio_)

pca = sklearn.decomposition.PCA(n_components=4)
pca.fit(X_train)
print("explained_variance_4:", pca.explained_variance_)
print("explained_variance_ratio_4:", pca.explained_variance_ratio_)
print("n:", pca.n_components_)
trans_data = pca.fit_transform(X_train)
print(trans_data.shape)


"""
    3.把原数据和降维结果合并，导出对比，可看出原数据值和降维后的结果值没有直接关系
"""
o = numpy.c_[X_train, trans_data]
writer = pandas.ExcelWriter('o.xlsx')
pandas.DataFrame(o).to_excel(writer)
writer.save()
