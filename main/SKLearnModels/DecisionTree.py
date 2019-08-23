"""
    0.前言：
    数据挖掘项目最繁琐且最不可自动化的环节在前期数据清洗，所以本程序不做演示和赘述。
    已经有一份处理好且自变量均有效的宽表数据dd_df.xlsx（见本目录下SAS数据文件）：
    loan_id为流水号，y是因变量（其中y=1表示坏客户），其余均为自变量。
    本程序着重演示如何使用决策树将连续数据离散化为最理想状态，并建模并展示模型效果。
"""
import pandas
import sklearn.tree
import pydotplus

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
    2. 开始决策树建模
"""
dtc = sklearn.tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None,
                                          min_samples_split=50, min_samples_leaf=10)
dtc.fit(X_train, y_train)
print('Train score:{:.3f}'.format(dtc.score(X_train, y_train)))
print('Test score:{:.3f}'.format(dtc.score(X_test, y_test)))
print('classes_:', dtc.classes_)
print('feature_importances_:', dtc.feature_importances_)
# print('tree_:', dtc.tree_)

"""
    3. 生成可视化图
"""
sklearn.tree.export_graphviz(dtc, out_file='tree.dot', feature_names=useful_columns.tolist(),
                             class_names=dtc.classes_, impurity=False, filled=True, rounded=True,
                             special_characters=True)
graph = pydotplus.graph_from_dot_file('tree.dot')
graph.write_pdf('tree.pdf')
print('Visible tree plot saved as pdf.')
