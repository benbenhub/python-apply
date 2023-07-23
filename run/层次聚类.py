import pandas as pd
import sys

path = str(sys.path[0]).replace('\\run','')

model_data = pd.read_csv(path + '/data/cities_10.csv', encoding='gbk')
data = model_data.loc[:, 'X1':]
# print(data.head())
# 主成分分析 进行中心标准化
from sklearn import preprocessing
data = preprocessing.scale(data)

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
newData = pca.fit(data)
print(pca.explained_variance_)
print(pca.explained_variance_ratio_)

from fa_kit import FactorAnalysis
from fa_kit import plotting as fa_plotting
fa = FactorAnalysis.load_data_samples(data, preproc_demean=True, preproc_scale=True)
fa.extract_components()

fa.find_comps_to_retain(method='top_n', num_keep=2)
# 通过最大方差法进行因子旋转
fa.rotate_components(method='varimax')
pd.DataFrame(fa.comps["rot"])

fa_scores = fa.get_component_scores(data)
fa_scores = pd.DataFrame()