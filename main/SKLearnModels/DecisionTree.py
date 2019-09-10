"""
    0.前言：
    数据挖掘项目最繁琐且最不可自动化的环节在前期数据清洗，所以本程序不做演示和赘述。
    已经有一份处理好且自变量均有效的宽表数据dd_df.xlsx（见本目录下SAS数据文件）：
    loan_id为流水号，y是因变量（二分类，其中y=1表示坏客户），其余均为自变量。
    本程序着重演示如何使用决策树方法建模，并评估模型。
"""
import pandas
import sklearn.tree
import numpy
from main.common.ConfusionMatrixModelEvaluation import ConfusionMatrixModelEvaluation

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

"""
    2. 开始决策树建模，主要（非全部）参数介绍：
        criterion：衡量节点“不纯度”的标准，默认gini，可选entropy信息增益；
        splitter：选择每个节点的分枝策略。
            默认best，表示根据criterion的最优表现选择自变量，需大量计算，一般用于训练样本数据量不大的场合
            可选random，表示随机划分属性，一般用于训练数据量较大的场合，可以减少计算量
        max_depth：设置决策树的最大深度，默认为None。None表示不对决策树的最大深度作约束，直到每个叶子结点上
            的样本均属于同一类，或者少于min_samples_leaf参数指定的叶子结点上的样本个数。一般不建议指定深度，
            过拟合或是降低复杂度的问题，还是通过剪枝来改善比较有效。
        min_samples_split：当对一个内部结点划分时，结点上的最小样本数，默认为2。
        min_samples_leaf：设置叶子结点上的最小样本数，默认为1。
        max_leaf_nodes: 设置决策树的最大叶子节点个数，默认为None，表示不加限制。
        min_impurity_decrease :打算划分一个内部结点时，只有当划分后不纯度减少值不小于该参数指定的值，才会对该结点进行划分，默认值为0。
        class_weight：设置样本数据中每个类的权重，默认为None，用户可以用字典型数据指定每个类的权重。
            该参数还可以设置为‘balance’，此时系统会按照输入的样本数据自动的计算每个类的权重，属于某个类的样本个数越多时，该类的权重越小。
"""
dtc = sklearn.tree.DecisionTreeClassifier(min_samples_split=100, min_samples_leaf=50)
dtc.fit(X_train, y_train)
print('Train score:{:.3f}'.format(dtc.score(X_train, y_train)))
print('Test score:{:.3f}'.format(dtc.score(X_test, y_test)))
print('classes_:', dtc.classes_)
useful_columns_df = pandas.DataFrame(useful_columns, columns=['column_name'])
useful_columns_df['feature_importance'] = dtc.feature_importances_
print(useful_columns_df)

"""
    3. 生成可视化图
"""
# sklearn.tree.export_graphviz(dtc, out_file='tree.dot', feature_names=useful_columns.tolist(),
#                              class_names=dtc.classes_, impurity=False, filled=True, rounded=True,
#                              special_characters=True)
# graph = pydotplus.graph_from_dot_file('tree.dot')
# graph.write_pdf('tree.pdf')
# print('Visible tree plot saved as pdf.')

"""
    4. 生成可视化图之后，可以再肉眼查看一下，做进一步的剪枝策略
"""

"""
    5.模型评估，依然可以用混淆矩阵方法，决策树模型的预测概率，就是叶子节点中目标类别的占比
"""
y_score = dtc.predict_proba(X_test)
y_test = pandas.Series.astype(y_test, 'int_')
result = numpy.c_[y_score[:, 1], y_test]
print(result)
cmme = ConfusionMatrixModelEvaluation(result)
cmme.allgraph()
