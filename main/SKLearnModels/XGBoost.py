"""
    0.前言：
    数据挖掘项目最繁琐且最不可自动化的环节在前期数据清洗，所以本程序不做演示和赘述。
    已经有一份处理好且自变量均有效的宽表数据dd_df.xlsx（见本目录下SAS数据文件）：
    loan_id为流水号，y是因变量（二分类，其中y=1表示坏客户），其余均为自变量。
    本程序着重演示如何使用XGBoost模型，并将结果可视化。
"""

import numpy
import pandas
import sklearn
import xgboost.sklearn
from matplotlib import pyplot
from main.common.ConfusionMatrixModelEvaluation import ConfusionMatrixModelEvaluation


"""
    1.导入数据，划分训练集和测试集数据
"""
data1 = pandas.read_excel('D:\\06-JerryTech\\dd_df.xlsx', sheet_name='Sheet1')
data1[['y']] = data1[['y']].astype('str')
useful_columns = pandas.Series(data1.columns)
useful_columns = useful_columns[useful_columns.values != 'loan_id']
useful_columns = useful_columns[useful_columns.values != 'y']
X_train, X_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(data1[useful_columns], data1['y'], test_size=0.2,
                                             random_state=0)


"""
    2. 开始XGBoost建模，XGBClassifier几个重要参数如下：
       1）learning_rate=0.1：学习率，步长
       2）n_estimators=1000：总共迭代的次数，即树的个数
       3）max_depth=6：树的最大深度
       4）min_child_weight=1：叶子节点最小权重，值越大，越容易欠拟合；值越小，越容易过拟合
       5）gamma=0：惩罚项系数，指定节点分裂所需的最小损失函数下降值。
       6）subsample=1：随机选择样本比例建立决策树，默认全部
       7）colsample_bytree=1：随机选择特征比例建立决策树，默认全部
       8）objective='binary:logistic'：指定目标函数，还可选：
                                       回归任务：reg:linear，reg:logistic 
                                       二分类：binary:logistic概率，binary：logitraw类别
                                       多分类：multi:softmax num_class=n 返回类别，multi:softprob num_class=n 返回概率
                                       rank:pairwise
       9）scale_pos_weight=1：解决样本个数不平衡的问题，正样本的权重。在二分类任务中，当正负样本比例失衡时，
                              设置正样本的权重，模型效果更好。例如，当正负样本比例为1:10时，scale_pos_weight=10
       10）silent=0：不输出中间过程，还可选1
       11）booster='gbtree'：树模型做为基分类器，还可选gbliner
       12）nthread=-1：使用全部cpu进行并行运算，还可选其他正整数
       13）early_stopping_rounds=None：在验证集上，当连续n次迭代，分数没有提高后，提前终止训练。
"""
xgbc = xgboost.sklearn.XGBClassifier(n_estimators=10)
xgbc.fit(X_train, y_train)
print(xgbc.score(X_train, y_train))
print(xgbc.score(X_test, y_test))
useful_columns_df = pandas.DataFrame(useful_columns, columns=['column_name'])
useful_columns_df['feature_importance'] = xgbc.feature_importances_
print(useful_columns_df)


"""
    3.混淆矩阵查看模型效果
"""
y_score = xgbc.predict_proba(X_test)
y_test = pandas.Series.astype(y_test, 'int_')
result = numpy.c_[y_score[:, 1], y_test]
print(result)
cmme = ConfusionMatrixModelEvaluation(result)
cmme.allgraph()


"""
    画出自变量重要性图示。
    另外想琢磨树的展示，可以展示单棵，但如何展示多棵并解读还未想明白。
"""
xgboost.plot_importance(xgbc)
pyplot.show()
xgboost.plot_tree(xgbc, num_trees=0)
fig = pyplot.gcf()
fig.set_size_inches(150, 100)
fig.savefig('tree0.png')
xgboost.plot_tree(xgbc, num_trees=3)
fig = pyplot.gcf()
fig.set_size_inches(150, 100)
fig.savefig('tree3.png')
