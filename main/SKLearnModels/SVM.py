"""
    0.前言：
    数据挖掘项目最繁琐且最不可自动化的环节在前期数据清洗，所以本程序不做演示和赘述。
    已经有一份处理好的宽表数据pf_data1（见本目录下SAS数据文件pf_data1.sas7bdat）：
    loan_id为流水号，y是因变量（其中y=1表示坏客户），其余均为自变量。
    本程序着重演示如何选取有效自变量、如何使用网格搜索方法寻找支持向量机的最优参数建模。
"""

import pandas
import math
import sas7bdat
import sklearn.svm
import matplotlib.pyplot as plt
from sklearn2pmml.pipeline import PMMLPipeline
from sklearn2pmml import sklearn2pmml

"""
    1.数据预处理：导入数据，计算每个变量的WOE值、IV值，筛选IV值>0.02的变量进入模型，并制作原始建模宽表
"""
sasdata = sas7bdat.SAS7BDAT('D:\\06-JerryTech\\pf_data1.sas7bdat', encoding="gb2312").to_data_frame()
sasdata['y'] = sasdata['y'].astype(int)

# 获取字段名称列表
columns = pandas.Series(sasdata.columns)
# 初始化WOE数据框
df_woe = pandas.DataFrame(columns=['column_name', 'column_value', 'bad', 'good',
                                   'total_bad', 'total_good', 'woe'])
# 计算总体好坏客户数
good_count = sasdata.query(' y == 0 ').count()[0]
bad_count = sasdata.query(' y == 1 ').count()[0]

# 计算每个字段值的woe值
i = 0
while i < columns.size:
    if columns[i] != 'loan_id' and columns[i] != 'y':
        tmpdf1 = sasdata.query(' y == 0 ').groupby(columns[i])['y'].count().reset_index()
        tmpdf1.columns = ['column_value', 'good']
        tmpdf2 = sasdata.query(' y == 1 ').groupby(columns[i])['y'].count().reset_index()
        tmpdf2.columns = ['column_value', 'bad']
        tmpdf = pandas.merge(tmpdf1, tmpdf2, on='column_value', how='left')
        df_woe = df_woe.append(tmpdf)
        df_woe['column_name'] = df_woe['column_name'].fillna(columns[i])
        df_woe['bad'] = df_woe['bad'].fillna(1)
        df_woe['good'] = df_woe['good'].fillna(0)
    i += 1

df_woe['total_bad'] = df_woe['total_bad'].fillna(bad_count)
df_woe['total_good'] = df_woe['total_good'].fillna(good_count)
df_woe['woe'] = df_woe.apply(lambda x: math.log((x.good/x.total_good)/(x.bad/x.total_bad)),
                             axis=1)
# 计算每个字段的IV值
df_woe['iv'] = df_woe.apply(lambda x: ((x.good/x.total_good)-(x.bad/x.total_bad))*x.woe, axis=1)
# df_woe.to_excel('D:\\06-JerryTech\\df_woe.xlsx', sheet_name='woe')
df_iv = df_woe.groupby('column_name')['iv'].sum().reset_index()
# df_iv.to_excel('D:\\06-JerryTech\\df_iv.xlsx', sheet_name='iv')

# 将IV值<0.02的字段挑出来，并据此删除原宽表
useless_columns = df_iv.query(' iv < 0.05 ')['column_name']
sasdata1 = sasdata.drop(useless_columns, axis=1)
# 根据业务实际考量，进一步删除意义重复的字段：
# registered_city、residential_city、yixin_loan、credit_card_cmt、credit_per_amt
sasdata1 = sasdata1.drop(['registered_city', 'residential_city', 'yixin_loan', 'credit_card_cmt',
                          'credit_per_amt'], axis=1)

# 合并宽表，将原始宽表中的字段值赋值为woe值
useful_columns = pandas.Series(sasdata1.columns)
useful_columns = useful_columns[useful_columns.values != 'loan_id']
useful_columns = useful_columns[useful_columns.values != 'y']
dd_df = sasdata1
for i in useful_columns.index:
    df_woe_tmp = df_woe[df_woe['column_name'] == useful_columns[i]]
    dd_df = pandas.merge(dd_df, df_woe_tmp, how='left',
                         left_on=useful_columns[i], right_on='column_value')
    dd_df['woe_'+useful_columns[i]] = dd_df['woe']
    dd_df.drop([useful_columns[i], 'column_name', 'column_value', 'iv', 'woe', 'good', 'bad',
                'total_good', 'total_bad'], axis=1, inplace=True)
# dd_df.to_excel('D:\\06-JerryTech\\dd_df.xlsx', sheet_name='widetable')
# 删除有空值的记录
dd_df = dd_df.dropna(axis=1)

"""
    2. 宽表已完成，划分训练集和测试集数据
"""
useful_columns = pandas.Series(dd_df.columns)
useful_columns = useful_columns[useful_columns.values != 'loan_id']
useful_columns = useful_columns[useful_columns.values != 'y']

X_train, X_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(dd_df[useful_columns], dd_df['y'], test_size=0.2,
                                             random_state=0)

"""
    3. 宽表已完成，开始支持向量机建模，模型的参数有3个比较重要，可以根据业务场景和具体数据调整达到最佳效果：
        1）kernel核函数：常用的核函数有linear（线性核函数）、poly（多项式核函数）、rbf（径像核函数/高斯核）、
           gmod（sigmod核函数）、precomputed（核矩阵）。在实际应用中，一般是先使用线性的kernel，如果效果
           不好再使用gaussian kernel（小的γ）和多项式kernel（小的Q）。不确定就用高斯核。
           在本案例中，用线性核效果反而比高斯核好。
        2）C惩罚参数cost：和松弛变量相关，默认值为1.0，表示错误项的惩罚系数。
           C越大，即对分错样本的惩罚程度越大，因此在训练样本中准确率越高，但是泛化能力降低；反之亦然。
        3）gamma：float参数，默认为auto，参数为核函数系数，只对rbf,poly,sigmod有效。
           如果gamma设置为auto，代表其值为样本特征数的倒数，即1/n_features。
           使用Gaussian SVM特别需要注意γ的使用，因为大的γ会导致SVM更复杂，也就更容易overfitting，所以一定要慎用！
"""
# 先用默认参数初始化模型
svc1 = sklearn.svm.SVC(C=1.6, kernel='rbf', gamma=1)
# 训练数据
svc1.fit(X_train, y_train)
# 查看训练效果，score函数在svm中就是正确分类的比例
print('train_score:', svc1.score(X=X_train, y=y_train))
print('test_score:', svc1.score(X=X_test, y=y_test))
# 输出预测结果
y = pandas.DataFrame(svc1.predict(X_test))
pred_data = X_test
pred_data['y_test'] = y_test
pred_data = pred_data.reset_index()
pred_data['y_predict'] = y
# 验证结果跟score是否一致，结果一致
print('人工验证准确率：', pred_data.query(' y_test==y_predict ').shape[0] / pred_data.shape[0])
print(pred_data.shape)
pred_data.to_excel('D:\\06-JerryTech\\pred_data.xlsx', sheet_name='pred_data')
"""
    4. 计算评价模型指标，画混淆矩阵：
       precision（精确率），recall（召回率），F-measure（F值）、accuracy（准确率）
       本次训练的结果比较坑，精确率、召回率、F值都为0，说明预测坏客户的能力比较差。
       个人感觉SVM模型，适合二分类样本比较均匀的场景，对于信用评分、反欺诈等要求召回率较高的场景不太适合
"""
precision = sklearn.metrics.precision_score(pred_data['y_test'], pred_data['y_predict'], pos_label=1)
print('precision（精确率）:', precision)
recall = sklearn.metrics.recall_score(pred_data['y_test'], pred_data['y_predict'], pos_label=1)
print('recall（召回率）:', recall)
f1 = sklearn.metrics.f1_score(pred_data['y_test'], pred_data['y_predict'], pos_label=1)
print('F-measure:', f1)
accuracy = sklearn.metrics.accuracy_score(pred_data['y_test'], pred_data['y_predict'])
print('accuracy（准确率）:', accuracy)

maxtrix = sklearn.metrics.confusion_matrix(pred_data['y_test'], pred_data['y_predict'])
plt.matshow(maxtrix)
plt.colorbar()
plt.xlabel('predict type')
plt.ylabel('true type')
plt.show()


"""
    方法二、用网格搜索方法寻找最优模型参数
"""
svc2 = sklearn.svm.SVC()
# 制定待训练的每个参数的若干备选值
param = {"C": [0.2, 0.4, 0.8, 1, 1.2, 1.6, 2],
         "kernel": ['rbf', 'linear'],
         "gamma": ['auto', 0.01, 0.05, 0.2, 0.4, 0.5, 1, 1.6, 2]}
gscv = sklearn.model_selection.GridSearchCV(svc2, param_grid=param, cv=3)
gscv = gscv.fit(X_train, y_train)
print("best score: {}".format(gscv.best_score_))
print("best params: {}".format(gscv.best_params_))


"""
    4.把模型导出至pmml文件，可被用于后续java系统，这里要提前装jdk貌似，暂时运行不了
"""
pipeline = PMMLPipeline([("classifier", svc1)])
pipeline.fit(X_train, y_train)
sklearn2pmml(pipeline, ".\\demo.pmml", with_repr=True)
