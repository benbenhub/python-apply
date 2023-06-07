import numpy as np
import pandas as pd
import sys

path = str(sys.path[0]).replace('\\run','')

data_raw = pd.read_excel(path + '/data/LR_practice.xlsx')

# print(data_raw.head())
# print(data_raw.info())

# 删除无用的变量
data_raw.drop(['id', 'Acc', 'edad2'], axis=1, inplace=True)
# 删除重复的数据
data_raw = data_raw.drop_duplicates()
# 查看数据缺失情况
# print(data_raw.isnull().mean())

# 对avg_exp直接填补
data_raw['avg_exp'] = data_raw['avg_exp'].fillna(data_raw['avg_exp'].mean())

# print(data_raw.isnull().mean())

# 对变量gender进行数据编码，并且对edu_class进行数据编码和缺失值填补
data_raw['gender'] = data_raw['gender'].map({'Male':1, 'Female':0})
label = data_raw['edu_class'].unique().tolist()
# print(label)
data_raw['edu_class'] = data_raw['edu_class'].apply(lambda x:label.index(x))
# print(data_raw.info())

# 异常值处理
from scipy import stats
z = stats.zscore(data_raw['Age'])
# print(z)
z_outlier = (z > 3) | (z < -3)
# print(z_outlier.tolist().index(1))
data_raw['Age'].iloc[40] = data_raw['Age'].drop(index=40).mean()

# 哑变量 构造数据特征
dummy = pd.get_dummies(data_raw['edu_class'], prefix='edu').iloc[:, 1:]
# print(dummy)
# 合并到原数据
data = pd.concat([data_raw, dummy], axis=1)
# print(data.head())
# 计算相关系数
# print(data[['avg_exp', 'gender', 'Ownrent', 'Selfempl', 'edu_class']].corr(method='kendall'))

# 散点图
import matplotlib.pyplot as plt
plt.scatter(data['avg_exp'], data['Income'])
# plt.show()

# 创建多元线性回归模型
from statsmodels.formula.api import ols
LR1 = 'avg_exp~gender+Age+Income+Ownrent+Selfempl+dist_home_val+dist_avg_income+edu_1+edu_2+edu_3+edu_4'
model = ols(LR1, data=data)
model = model.fit()
print(model.summary())