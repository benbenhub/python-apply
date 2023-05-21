import pandas as pd
import os
import numpy as np
import sys

path = str(sys.path[0]).replace('\\run','')
camp = pd.read_csv(path + '/data/teleco_camp_orig.csv')
# 脏数据或数据不正确
import matplotlib.pyplot as plt

plt.hist(camp['AvgIncome'], bins=20, normed=True)
# 查看人均收入分布情况
# plt.show()
camp['AvgIncome'].describe(include='all')

plt.hist(camp['AvgHomeValue'], bins=20, normed=True)
camp['AvgHomeValue'].describe(include='all')
plt.show()
# 这里的0值应该是缺失值，处理0值
camp['AvgIncome'] = camp['AvgIncome'].replace({0: np.NaN})
# print(camp.AvgIncome.min())
# print(camp.AvgIncome.max())
# 由于数据中存在缺失值，因此需要指定绘图的值域
plt.hist(camp['AvgIncome'], bins=20, normed=True, range=(camp.AvgIncome.min(), camp.AvgIncome.max()))
# 处理0值后的AvgHomeValue
# plt.show()

camp['AvgIncome'].describe(include='all')

camp['AvgHomeValue'] = camp['AvgHomeValue'].replace({0: np.NaN})

plt.hist(camp['AvgHomeValue'], bins=20, normed=True, range=(camp.AvgHomeValue.min(), camp.AvgHomeValue.max()))

camp['AvgHomeValue'].describe(include='all')


# 用盖帽法处理离群值
def blk(floor, root):
    def f(x):
        if x < floor:
            x = floor
        elif x > floor:
            x = root
        return x
    return f

q1 = camp['Age'].quantile(0.01) # 计算百分位数
q99 = camp['Age'].quantile(0.99)
blk_tot = blk(floor=q1, root=q99)
camp['Age'] = camp['Age'].map(blk_tot)
print(camp['Age'].describe())





