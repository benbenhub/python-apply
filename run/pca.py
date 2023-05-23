import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

path = str(sys.path[0]).replace('\\run','')
profile = pd.read_csv(path + '/data/profile_telecom.csv')

print(profile.head())
# 分析变量的相关关系
data = profile.loc[:, 'cnt_call':]
print(data.corr(method='pearson'))
# 数据标准化
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
# 数据标准化函数，默认进行中心标准化
data_scaled = scale(data)
# 
pca = PCA(n_components=2, whiten=True).fit(data_scaled)

print(pca.explained_variance_ratio_)
print(pca.components_)
print(pca.transform(data_scaled))