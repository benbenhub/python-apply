import pandas as pd
import sys

path = str(sys.path[0]).replace('\\run','')

model_data = pd.read_csv(path + '/data/profile_bank.csv')
data = model_data.loc[ :, 'CNT_TBM':'CNT_CSC']
# print(data.head())

# 查看相关系数矩阵，判定做变量降维的必要性
corr_matrix = data.corr(method='pearson')
# corr_matrix = corr_matrix.abs()
# print(corr_matrix)

# 主成分分析
from sklearn import preprocessing
data = preprocessing.scale(data)
# 判断保留主成分的数量
from sklearn.decomposition import PCA
'''
1、第一次设的n_con
'''
pca = PCA(n_components=3)
newData = pca.fit(data)
# print(pca.explained_variance_)
# print(pca.explained_variance_ratio_)

pd.DataFrame(pca.components_).T

from fa_kit import FactorAnalysis
from fa_kit import plotting as fa_plotting
fa = FactorAnalysis.load_data_samples(data, preproc_demean=True, preproc_scale=True)
fa.extract_components()

fa.find_comps_to_retain(method='top_n', num_keep=3)
fa.rotate_components(method='varimax')
fa_plotting.graph_summary(fa)
# 获得因子得分
score = pd.DataFrame(fa.comps["rot"])
fa_scores = score.rename(columns={0: "ATM_POS", 1: "TBM", 2: "CSC"})

# 建立聚类模型
# 查看变量的偏度
var = ["ATM_POS", "TBM", "CSC"]
skew_var = {}
for i in var:
    skew_var[i] = abs(fa_scores[i].skew())
    skew = pd.Series(skew_var).sort_values(ascending=False)
# print(skew)

# 进行k-means聚类
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3)
result = kmeans.fit(fa_scores)
# print(result)

# 对分类结果进行排序
model_data_l = model_data.join(pd.DataFrame(result.labels_))
model_data_l = model_data_l.rename(columns={0: "clustor"})
print(model_data_l.head())

# import matplotlib 
# get_ipython().magic('matplotlib inline')
model_data_l.clustor.value_counts().plot(kind='pie')

# 进行变量的正态分布转换
import numpy as np
from sklearn import preprocessing
quantile_transformer = preprocessing.QuantileTransformer(output_distribution='normal', random_state=0)
fa_scores_trans = quantile_transformer.fit_transform(fa_scores)
fa_scores_trans = pd.DataFrame(fa_scores_trans)
fa_scores_trans = fa_scores_trans.rename(columns={0: "ATM_POS", 1: "TBM", 2: "CSC"})
# print(fa_scores_trans.head())

var = ["ATM_POS", "TBM", "CSC"]
skew_var = {}
for i in var:
    skew_var[i] = abs(fa_scores_trans[i].skew())
    skew = pd.Series(skew_var).sort_values(ascending=False)
# print(skew)

# 使用k-means聚类算法
kmeans = KMeans(n_clusters=3)
result = kmeans.fit(fa_scores)
model_data_l = model_data.join(pd.DataFrame(result.labels_))
model_data_l = model_data_l.rename(columns={0: "clustor"})
print(model_data_l.head())

# import matplotlib 
# get_ipython().magic('matplotlib inline')
model_data_l.clustor.value_counts().plot(kind='pie')