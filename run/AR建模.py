import pandas as pd
import sys
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import statsmodels.tsa.api as smt
from statsmodels.tsa.stattools import adfuller   #平稳性检验
from statsmodels.stats.diagnostic import acorr_ljungbox   #白噪声检验
import numpy as np
from statsmodels.graphics.tsaplots import plot_predict

path = str(sys.path[0]).replace('\\run','')

df = pd.read_csv(path + '/data/time_series_1.csv', index_col=0)
# 将原始数据转换为时间序列数据
index = pd.date_range(start='2019-01-01', periods=1000, freq='D')
df.index = index
df.plot()

# 数据是非平稳的，因此需要进行差分
ts_diff = df.diff(1)
ts_diff = ts_diff[1:]
plt.plot(ts_diff)
plt.xlabel('Time')
# plt.show()

# 定阶
# 自相关图和偏自相关图，默认阶数为30阶
def draw_acf_pacf(ts, subtitle, lags=30):
    print("自相关图和偏自相关图,maxlags={}".format(lags))
    f = plt.figure(facecolor='white', figsize=(18,4))
    ax1 = f.add_subplot(121)
    plot_acf(ts, lags=lags, ax=ax1, title='ACF\n{}'.format(subtitle))
    ax2 = f.add_subplot(122)
    plot_pacf(ts, lags=lags, ax=ax2, title='PACF\n{}'.format(subtitle))
    # plt.show()

draw_acf_pacf(ts_diff, 'ts - 一阶差分后', 30)
# 函数AROMA（）可以建立AR模型 MA模型 ARMA模型 带分差的ARIMA模型
arima_1 = smt.ARIMA(df, order=(1, 1, 0)).fit()
print(arima_1.summary())

# 残差白噪声检验
def randomness(ts, lags=31):
    rdtest = acorr_ljungbox(ts, lags=lags)
    rddata = np.c_[range(1, lags+1), rdtest['lb_pvalue']]
    rdoutput = pd.DataFrame(rddata, columns=['lags', 'p-value'])
    return rdoutput.set_index('lags')

print(randomness(arima_1.resid, 5))

# AR模型预测
fig, ax = plt.subplots(figsize=(10, 8))
fig = plot_predict(arima_1, start="2019-01-02", end="2021-10-26", ax=ax)
legend = ax.legend(loc="upper left")
print(legend)
plt.show()