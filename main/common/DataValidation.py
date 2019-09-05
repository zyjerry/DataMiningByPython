"""
    0.前言：
    本程序详细演示二分类型数据预处理的主要方法，包括变量相关性的验证、显著性筛选等。
    已经有一份清洗好且自变量均有效的宽表数据pf_data1.xlsx：
    loan_id为流水号，y是因变量（其中y=1为正例，表示坏客户），其余均为自变量，大部分是离散值，自变量之间可能有相关性。
"""
import numpy
import pandas
import math
import sklearn.linear_model
from sklearn.model_selection import KFold
import scipy.stats
import sklearn.feature_selection

class DataValidation:
    """    私有属性声明    """
    # 原始的完整数据
    __dataframe = pandas.DataFrame()
    # 转换成woe的数据
    __dataframe_woe = pandas.DataFrame()
    # 流水号字段名
    __id = ''
    # 目标变量名称
    __y = ''
    # 自变量名称
    __x = pandas.Series()
    # 数据摘要
    __dataabstract = pandas.DataFrame()
    # 各变量的woe值
    __column_woe = pandas.DataFrame()
    # 各变量的iv值
    __column_iv = pandas.DataFrame()

    """
        构造函数，传入待评估数据文件
        参数：
            filetype：文件类型，目前只支持excel
            filename：文件名
            sheetname：excel中的sheet名
            id：数据的流水号
            y：数据的目标变量，除了id和y之外，其他变量默认全部是待分析的自变量
    """
    def __init__(self, filetype='xlsx', filename='', sheetname='Sheet1', id='id', y='y'):
        self.__dataframe = pandas.read_excel(filename, sheet_name=sheetname)
        self.__id = id
        self.__y = y
        self.__x = pandas.Series(self.__dataframe.columns)
        self.__x = self.__x[self.__x != self.__id]
        self.__x = self.__x[self.__x != self.__y]

    """    
        当print该类对象时返回的内容    
    """
    def __str__(self):
        printstr = "DataValidation类：\n初始数据规模："
        printstr += str(self.__dataframe.shape[0]) + "行，"
        printstr += str(self.__dataframe.shape[1]) + "列\n"
        printstr += "列名：\n" + str(pandas.Series(self.__dataframe.columns))
        return printstr

    """
        返回数据
    """
    def getdata(self):
        return self.__dataframe, self.__x, self.__y

    """
        数据探查，显示数据的概要，保存，包括：
        1、哪些自变量是离散型、哪些自变量是连续性
        2、离散型自变量的取值范围、频数
        3、连续性自变量的统计指标：最大值、最小值、平均值、中位数、1/4分位数、3/4分位数
        4、因变量的值和分布
        返回：
            df_abs：变量总体概要，类型、不同值的数量
            df_abs_discrete：离散值的分布
            df_abs_conti：连续值的分布
            df_abs_y：因变量的分布
    """
    def abstract(self, filename='abstract.xlsx'):
        # 统计y变量的分布
        df_abs_y = pandas.DataFrame()
        y = pandas.DataFrame(self.__dataframe.groupby(self.__y)[self.__y].count())
        y.columns = ['y_count']
        y = y.reset_index()
        df_abs_y = df_abs_y.append(y, ignore_index=True)
        # 统计各自变量的分布
        df_abs = pandas.DataFrame()    # 变量总体概要，类型、不同值的数量
        # 离散值的分布
        df_abs_discrete = pandas.DataFrame(columns=['column_name', 'column_value', 'column_count'])
        # 连续值的分布
        df_abs_conti = pandas.DataFrame()
        for i in self.__x:
            # 字符型变量默认为离散值，统计各值频数
            if self.__dataframe[i].dtype == 'object':
                ddd = pandas.DataFrame(self.__dataframe.groupby(i)[i].count())
                ddd.columns = ['column_count']
                ddd['column_value'] = ddd.index
                ddd['column_name'] = i
                df_abs_discrete = df_abs_discrete.append(ddd, ignore_index=True)
                df_abs_discrete['column_name'].fillna(i)
                a = [[i, self.__dataframe[i].dtype, self.__dataframe.groupby(i)[i].count().shape[0]]]
                df_abs = df_abs.append(a, ignore_index=True)
            # 其他变量默认为连续值，统计主要分布指标
            if self.__dataframe[i].dtype != 'object':
                s1 = pandas.Series(self.__dataframe[i])
                d1 = pandas.DataFrame(s1.describe())
                df_abs_conti = df_abs_conti.append(d1.T.reset_index())
                b = [[i, self.__dataframe[i].dtype, 0]]
                df_abs = df_abs.append(b, ignore_index=True)
        df_abs.columns = ['column_name', 'column_type', 'column_distinct_count']
        df_abs_conti.columns = ['column_name', 'count', 'mean', 'std', 'min', 'p_25%',
                                  'p_50%', 'p_75%', 'max']
        # 把上述统计信息写入到excel表中
        writer = pandas.ExcelWriter(filename)
        df_abs.to_excel(writer, sheet_name='columntype')
        df_abs_discrete.to_excel(writer, sheet_name='discretecolumns')
        df_abs_conti.to_excel(writer, sheet_name='continuouscolumns')
        df_abs_y.to_excel(writer, sheet_name='y')
        writer.save()
        return df_abs, df_abs_discrete, df_abs_conti, df_abs_y

    """
        计算某一个离散变量的woe值和IV，并通过dataframe的形式返回
        参数：
            columnname：变量名
        返回：
            一个pandas的dataframe，包含4列：变量名、变量值、y=1数量、y=2数量、woe值、IV值、IV总值
    """
    def calcwoe(self, columnname):
        df_woe = pandas.DataFrame(columns=['column_name', 'column_value', 'bad', 'good',
                                           'total_bad', 'total_good', 'woe', 'iv', 'iv_total'])
        tmpdf1 = self.__dataframe.query(self.__y + ' == 0 ').groupby(columnname)[self.__y].count().reset_index()
        tmpdf1.columns = ['column_value', 'good']
        tmpdf2 = self.__dataframe.query(self.__y + ' == 1 ').groupby(columnname)[self.__y].count().reset_index()
        tmpdf2.columns = ['column_value', 'bad']
        tmpdf = pandas.merge(tmpdf1, tmpdf2, on='column_value', how='left')
        df_woe = df_woe.append(tmpdf)
        df_woe['column_name'] = df_woe['column_name'].fillna(columnname)
        df_woe['bad'] = df_woe['bad'].fillna(1)
        df_woe['good'] = df_woe['good'].fillna(0)

        good_num = self.__dataframe.query('y == 0 ').count()[0]
        bad_num = self.__dataframe.query('y == 1 ').count()[0]
        df_woe['total_bad'] = df_woe['total_bad'].fillna(bad_num)
        df_woe['total_good'] = df_woe['total_good'].fillna(good_num)
        df_woe['woe'] = df_woe.apply(
            lambda x: math.log((x.good / x.total_good) / (x.bad / x.total_bad)), axis=1)
        df_woe['iv'] = df_woe.apply(
            lambda x: ((x.good / x.total_good) - (x.bad / x.total_bad)) * x.woe, axis=1)
        df_iv = df_woe['iv'].sum()
        df_woe['iv_total'] = df_woe['iv_total'].fillna(df_iv)
        return df_woe

    """
        计算所有变量的WOE、IV值，并汇总入DataFrame中返回
    """
    def calcallwoesivs(self, filename='woeivs.xlsx'):
        for i in self.__x:
            self.__column_woe = self.__column_woe.append(self.calcwoe(i))
            self.__column_iv = self.__column_iv.append(self.calcwoe(i).groupby('column_name')['iv'].sum().reset_index())
        writer = pandas.ExcelWriter(filename)
        self.__column_woe.to_excel(writer, sheet_name='woe')
        self.__column_iv.to_excel(writer, sheet_name='iv')
        writer.save()
        return self.__column_woe, self.__column_iv

    """
        计算某两个离散变量之间的相关显著性，使用卡方检验法
        参数：
            x1_discrete：离散变量名
            x2_discrete：离散变量名
        返回：卡方分析的参数值
    """
    def calccorrebychi2(self, x1_discrete, x2_discrete):
        contingency_table = pandas.crosstab(
            self.__dataframe[x1_discrete], self.__dataframe[x2_discrete]
        )
        f_obs = numpy.array(contingency_table)
        chi2, p, dof, expected = scipy.stats.chi2_contingency(f_obs)

        return chi2, p, dof, expected

    """
        计算一个连续单变量和一个离散单变量的显著性，使用方差分析方法
        参数：
            x1_continuous：连续变量名
            x2_discrete：离散变量名
        返回：
            f：F统计量
            p：推翻原假设的概率，通常<0.05的话说明变量之间有相关性
    """
    def calccorrebyanova(self, x1_continuous, x2_discrete):
        # print(x1_continuous, ',  ', x2_discrete)
        grouped = self.__dataframe.groupby(x2_discrete)[x1_continuous]
        f, p = scipy.stats.f_oneway(*[v for k, v in grouped])
        return f, p

    """
        计算两个连续变量的相关性，用相关系数
    """
    def calccorrebycoeff(self, x1_continuous, x2_continuous):
        u1 = numpy.array(self.__dataframe[x1_continuous])
        u2 = numpy.array(self.__dataframe[x2_continuous])
        c = numpy.corrcoef(u1, u2)
        return c[0, 1]

    """
        计算某两个变量的相关性
        参数：
            x1：变量名
            x2：变量名
        返回：
            若是卡方检验或方差分析，返回p值，p值<0.05说明强相关
            若是相关系数，越接近1或-1说明强相关
    """
    def calccorr(self, x1, x2):
        chi2 = 0.0
        f = 0.0
        p = 0.0
        c = 0.0
        # 如果x1是离散型、x2是离散型，使用卡方检验
        if self.__dataframe[x1].dtype == 'object' and self.__dataframe[x2].dtype == 'object':
            chi2, p, dof, expected = self.calccorrebychi2(x1,x2)
        # 如果x1是离散型、x2是连续型，使用方差分析
        if self.__dataframe[x1].dtype == 'object' and self.__dataframe[x2].dtype != 'object':
            f, p = self.calccorrebyanova(x2, x1)
        # 如果x1是连续型、x2是离散型，使用方差分析
        if self.__dataframe[x1].dtype != 'object' and self.__dataframe[x2].dtype == 'object':
            f, p = self.calccorrebyanova(x1, x2)
        # 如果x1是连续型、x2是连续型，使用Pearson相关系数
        if self.__dataframe[x1].dtype != 'object' and self.__dataframe[x2].dtype != 'object':
            c = self.calccorrebycoeff(x1, x2)
        return chi2, f, p, c

    """
        计算所有自变量之间的相关性
        返回一张DataFrame表
        实验证明对于离散变量值较多的变量，两两之间的卡方检验很容易得出高度相关性，不可靠
        个人推测是因为样本数量、各自独立性、分布形态不符合卡方检验的前提条件
    """
    def calcallcorrs(self, filename='corr.xlsx'):
        df_corr = pandas.DataFrame()
        for i in self.__x:
            for j in self.__x:
                if i != j:
                    chi2, f, p, c = self.calccorr(i, j)
                    s = pandas.Series([i, j, chi2, f, p, c])
                    df_corr = df_corr.append(s, ignore_index=True)
        df_corr.columns = ['column_1', 'column_2', 'chi2', 'f', 'p', 'c']
        writer = pandas.ExcelWriter(filename)
        df_corr.to_excel(writer, sheet_name='corr')
        writer.save()
        return df_corr

    """
        将原始数据转换成woe数据
    """
    def datatowoe(self, filename='pf_data_woe.xlsx'):
        self.__dataframe_woe = self.__dataframe
        for i in self.__x:
            df_woe_tmp = self.__column_woe[self.__column_woe['column_name'] == i]
            self.__dataframe_woe = pandas.merge(self.__dataframe_woe, df_woe_tmp, how='left',
                                 left_on=i, right_on='column_value')
            self.__dataframe_woe['woe_' + i] = self.__dataframe_woe['woe']
            self.__dataframe_woe.drop([i, 'column_name', 'column_value', 'iv', 'woe', 'good', 'bad',
                 'total_good', 'total_bad','iv_total'], axis=1, inplace=True)
        self.__dataframe_woe.to_excel(filename, sheet_name='Sheet1', index=False)
        # 删除有空值的记录
        self.__dataframe_woe = self.__dataframe_woe.dropna(axis=1)
        return self.__dataframe_woe

"""    当直接运行本类时，打印默认初始化值，不做任何计算    """
if __name__ == "__main__":
    dv1 = DataValidation(filetype='xlsx', filename='pf_data1.xlsx', sheetname='Sheet1',
                        id='loan_id', y='y')
    dv1.calcallwoesivs()
    dv1.calcallcorrs()
    dv1.datatowoe()

    dv2 = DataValidation(filetype='xlsx', filename='pf_data_woe.xlsx', sheetname='widetable',
                        id='loan_id', y='y')
    dv2.abstract()
    data, x, y = dv2.getdata()
    corr = data[x].corr()
    corr.to_excel('aaa.xlsx', sheet_name='corr')
    print(data[x].corr())
    dv2.calcallwoesivs()
    dv2.calcallcorrs(filename='corr2.xlsx')
