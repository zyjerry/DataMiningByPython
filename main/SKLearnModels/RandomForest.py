"""
    0.前言：
    数据挖掘项目最繁琐且最不可自动化的环节在前期数据清洗，所以本程序不做演示和赘述。
    已经有一份处理好且自变量均有效的宽表数据dd_df.xlsx（见本目录下SAS数据文件）：
    loan_id为流水号，y是因变量（二分类，其中y=1表示坏客户），其余均为自变量。
    本程序着重演示如何使用K折交叉验证方法寻找随机森林的最简最优模型。
"""

import pandas
import sklearn.ensemble
import pydotplus
import sklearn.model_selection

"""
    1.导入数据，使用3折交叉验证，划分数据，每两份训练，一份测试
"""
data1 = pandas.read_excel('D:\\06-JerryTech\\dd_df.xlsx', sheet_name='Sheet1')
data1[['y']] = data1[['y']].astype('str')
useful_columns = pandas.Series(data1.columns)
useful_columns = useful_columns[useful_columns.values != 'loan_id']
x_columns = useful_columns[useful_columns.values != 'y']
kfold = sklearn.model_selection.KFold(n_splits=3, shuffle=True)

"""
    2.对每一轮划分的数据，进行随机森林的建模
"""
i = 0
for train, test in kfold.split(data1[useful_columns]):
    X_train = data1.loc[train.tolist()][x_columns]
    X_test = data1.loc[test.tolist()][x_columns]
    y_train = data1.loc[train.tolist()]['y']
    y_test = data1.loc[test.tolist()]['y']

    """
        2.1 开始RF建模，RF有几个参数比较重要：
            1）n_estimators：integer，optional（default = 10） 森林里的树木数量。
            2）criteria：string，可选（default =“gini”）分割特征的测量方法 
            3）max_depth：integer或None，可选（默认=无）树的最大深度
            4）bootstrap：boolean，optional（default =  True）是否在构建树时使用自举样本。
    """
    rfc = sklearn.ensemble.RandomForestClassifier(n_estimators=8, max_depth=8,
                                                  min_samples_split=64, min_samples_leaf=32)
    rfc.fit(X_train, y_train)
    print(rfc.score(X_train, y_train))
    print(rfc.score(X_test, y_test))
    print(rfc.classes_)
    columns_importance = pandas.DataFrame(x_columns, columns=['column_name'])
    columns_importance['feature_importance'] = rfc.feature_importances_
    print(columns_importance)

    """
        2.2 将模型结果转存为图
    """
    for index, model in enumerate(rfc.estimators_):
        filename = 'forest_' + str(i) + '_' + str(index) + '.pdf'
        dot_data = sklearn.tree.export_graphviz(model, out_file=None,
                             feature_names=x_columns.tolist(),
                             filled=True, rounded=True,
                             special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data)
        # 使用ipython的终端jupyter notebook显示。
        graph.write_pdf(filename)
    i += 1

"""
    3.综合评估3次模型score差不多，较为稳定，可以酌情减少树的棵树和深度，重新尝试
    不怕麻烦的话可以内部再套个循环对比，我就不写了
"""
