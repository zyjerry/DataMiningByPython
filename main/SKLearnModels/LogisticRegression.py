"""
    0.前言：
    本程序详细演示逻辑回归在信用评分卡中的建模过程和关键技术点，以及模型评估。
    这是一个典型的二元分类回归问题。
    数据挖掘项目最繁琐且最不可自动化的环节在前期数据清洗和验证，所以本程序不做演示和赘述，验证细节请移步DataValidation.py。
    已经有一份处理好且自变量均有效的宽表数据dd_df.xlsx：
    loan_id为流水号，y是因变量（其中y=1为正例，表示坏客户），其余均为自变量换算成的woe值。
"""
import numpy
import pandas
import sklearn
import sklearn.linear_model
import sklearn.model_selection
import matplotlib.pyplot as plt
from main.common.ConfusionMatrixModelEvaluation import ConfusionMatrixModelEvaluation

"""
    1.导入数据，划分训练集和测试集数据
"""
data1 = pandas.read_excel('dd_df.xlsx', sheet_name='Sheet1')
useful_columns = pandas.Series(data1.columns)
useful_columns = useful_columns[useful_columns.values != 'loan_id']
useful_columns = useful_columns[useful_columns.values != 'y']
X_train, X_test, y_train, y_test = \
    sklearn.model_selection.train_test_split(data1[useful_columns], data1['y'], test_size=0.2,
                                             random_state=0)

"""
    2. 开始逻辑回归建模，模型的参数有3个比较重要，可以根据业务场景和具体数据调整达到最佳效果：
        1)penalty（正则化参数）：用来指定惩罚的基准，字符串‘l1’或‘l2’,默认‘l2’。
          如果选择‘l1’，solver参数就只能用‘liblinear’算法；
          如果选择‘l2’，solver参数可以选择‘liblinear’、‘newton-cg’、‘sag’和‘lbfgs’这四种算法。
        2）C惩罚参数cost：即正则化系数λ的倒数，必须为正数，默认为1。和SVM中的C一样，值越小，代表正则化越强。
        3）solver: solver参数决定了我们对逻辑回归损失函数的优化方法，有四种算法可以选择：
           a) liblinear：使用了开源的liblinear库实现，内部使用了坐标轴下降法来迭代优化损失函数。
           b) lbfgs：拟牛顿法的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
           c) newton-cg：也是牛顿法家族的一种，利用损失函数二阶导数矩阵即海森矩阵来迭代优化损失函数。
           d) sag：即随机平均梯度下降，是梯度下降法的变种，和普通梯度下降法的区别是每次迭代仅仅用一部分的样本来计算梯度，适合于样本数据多的时候。       
           从上面的描述可以看出，newton-cg、lbfgs和sag这三种优化算法时都需要损失函数的一阶或者二阶连续导数，
           因此不能用于没有连续导数的L1正则化，只能用于L2正则化。而liblinear通吃L1正则化和L2正则化。
           同时，sag每次仅仅使用了部分样本进行梯度迭代，所以当样本量少的时候不要选择它，而如果样本量非常大，比如大于10万，sag是第一选择。
           但是sag不能用于L1正则化。所以当你有大量的样本，又需要L1正则化的话就要自己做取舍了。要么通过对样本采样来降低样本量，要么回到L2正则化。
           但是liblinear也有自己的弱点！我们知道，逻辑回归有二元逻辑回归和多元逻辑回归。
           对于多元逻辑回归常见的有one-vs-rest(OvR)和many-vs-many( MvM)两种。
           而MvM一般比OvR分类相对准确一些。而liblinear只支持OvR，不支持MvM。
           这样如果我们需要相对精确的多元逻辑回归时，就不能选择liblinear了。也意味着如果我们需要相对精确的多元逻辑回归不能使用L1正则化了。
"""
lr = sklearn.linear_model.LogisticRegression(penalty='l1', C=1, solver='liblinear')
lr.fit(X_train, y_train)
y_predict = lr.predict_proba(X_test)
print('score:', lr.score(X_test, y_test))
print('coef:', lr.coef_)
print('intercept:', lr.intercept_)

"""
    3.根据结果画LIFT曲线、ROC曲线、KS曲线、计算AUC值
      sklearn.metrics.roc_curve、auc函数能够自动根据实际值和预测概率计算结果，
      但本程序未使用，自行封装了ConfusionMatrixModelEvaluation类直接调用
"""
# 画LIFT图
result = numpy.dstack((y_predict[:, 1], y_test))[0]
cmme = ConfusionMatrixModelEvaluation(result)
print(cmme)
cmme.liftgraph(divisiontype="percentage")

# 画ROC曲线和洛伦兹曲线
cmme.roclorenzgraph(divisiontype="percentage")

# 画KS曲线
cmme.ksgraph(divisiontype="percentage")

# 获取结果指标明细并导出为excel
indicators = cmme.getindicators(divisiontype="percentage")
df_indicators = pandas.DataFrame(indicators)
df_indicators.columns = ['分位点', '召回率', '查准率', '特指度', '准确性', 'F-measure']
writer = pandas.ExcelWriter('indicators.xlsx')
df_indicators.to_excel(writer)
writer.save()
