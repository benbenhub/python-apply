import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import sys

path = str(sys.path[0]).replace('\\run','')
df = pd.read_csv(path + '/data/city_distance.csv', skipinitialspace=True)

df_filled = df.fillna(0)
distance_array = np.array(df_filled.iloc[:, 1:])
cities = distance_array + distance_array.T

# print(cities)
from sklearn.manifold import MDS

mds = MDS(n_components=2, dissimilarity='precomputed', random_state=123)
mds.fit_transform(cities)
# print(mds.stress_)
# print(mds.embedding_)

import seaborn
seaborn.set()
x = mds.embedding_[:, 0]
y = mds.embedding_[:, 1]
plt.scatter(x, y,)
for a, b, s in zip(x, y, df['City']):
    plt.text(a, b, s, fontsize=12)

plt.show()
