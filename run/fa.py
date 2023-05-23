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
pca = PCA(n_components=9)
pca.fit(data)

print(pca.explained_variance_)
print(pca.explained_variance_ratio_)

from fa_kit import FactorAnalysis
from fa_kit import plotting as fa_plotting
fa = FactorAnalysis.load_data_samples(data, preproc_demean=True, preproc_scale=True)
fa.extract_components()

fa.find_comps_to_retain(method='top_n', num_keep=2)

fa.rotate_components(method='varimax')
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
