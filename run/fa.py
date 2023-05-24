import pandas as pd 
import sys

path = str(sys.path[0]).replace('\\run','')
model_data = pd.read_csv(path + '/data/cities_10.csv', encoding='gbk')
citi10_fa = pd.read_csv(path + '/data/citi10_fa.csv')

data = model_data.loc[:, 'X1':]

from sklearn import preprocessing
data = preprocessing.scale(data)
# print(data)
from sklearn.decomposition import PCA
# 主成分分析
pca = PCA(n_components=9)
pca.fit(data)

print(pca.explained_variance_)
print(pca.explained_variance_ratio_)
# 数据结果：保留两个主成分，累计解释的数据变异率达到92%

from fa_kit import FactorAnalysis
from fa_kit import plotting as fa_plotting
# 加载数据
fa = FactorAnalysis.load_data_samples(data, preproc_demean=True, preproc_scale=True)
# 抽取主成分
fa.extract_components()
# 确定保留因子个数
fa.find_comps_to_retain(method='top_n', num_keep=2)
# 通过最大方差进行因子旋转
fa.rotate_components(method='varimax')
# 查看因子权重
pd.DataFrame(fa.comps["rot"])

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
x = citi10_fa['Gross']
y = citi10_fa['Avg']
label = citi10_fa['AREA']
plt.scatter(x, y)
for a,b,l in zip(x, y, label):
    plt.text(a, b+0.1, '%s.' % l, ha='center', va='bottom', fontsize=14)

plt.show()
