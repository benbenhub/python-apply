import os
import numpy as np
from scipy import stats
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import sys

path = str(sys.path[0]).replace('\\run','')

churn = pd.read_csv(path + '/data/telecom_churn.csv')
# print(churn.head())
# print(churn.info())
# 数据没有缺失情况

# 相关关系分析 
# 列联表
cross_table = pd.crosstab(churn.posTrend, churn.churn, margins=True)
# print(cross_table)
def percConvert(ser):
    return ser/float(ser[-1])
cross_table.apply(percConvert, axis=1)
# 列联表的卡方独立性检验：若显著，则说明二者存在相关关系
print('''chisq = %6.4f
p-value = %6.4f
dof = %i
expected_frep = %s''' %stats.chi2_contingency(cross_table.iloc[:2, :2]))
# 随机抽样 建立训练集与测试集
train = churn.sample(frac=0.7, random_state=666)
test = churn[~ churn.index.isin(train.index)]

print(' 训练集样本量: %i \n 测试样本量: %i' %(len(train), len(test)))

# 建立一元逻辑回归模型
lg = smf.logit('churn ~ duration', train).fit()
# 使用summary函数查看模型的一些信息
# print(lg.summary())

# 变量选择
# 向前回归法s