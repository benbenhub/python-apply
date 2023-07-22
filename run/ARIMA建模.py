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

# 现有某公司电脑产品销量的数据，想要对电商渠道的销量建立模型并进行预测
call = pd.read_csv(path + '/data/Call.csv')
# print(call.head())
# 将数据集分成建模数据集与验证数据集。建模数据集为前70%，验证数据集后30%
train_call = call.loc[0:183, ['starting_date', 'dotcom_calls']]
test_call = call.loc[184:262, ['starting_date', 'dotcom_calls']]
print("训练集数：", len(train_call))
print("验证集数：", len(test_call))

# 绘制训练数据的时序图，判断序列的平稳性
train_call['starting_date'] = pd.to_datetime(train_call['starting_date'])
train_call.set_index("starting_date", inplace=True)
test_call['starting_date'] = pd.to_datetime(test_call['starting_date'])
test_call.set_index("starting_date", inplace=True)
# 训练数据时序图
# plt.figure(figsize=(20, 6))
# plt.plot(train_call)
# plt.xlabel('Time')
# plt.show()

# 52步的差分，去除周期趋势
call_1 = train_call.diff(52)

call_1 = call_1[52:184]
# plt.plot(call_1)
# plt.xlabel('Time')
# plt.show()

# 时序图仍不稳定，继续做1阶差分消除长期趋势
call_2 = call_1.diff(1)
# plt.plot(call_2)
# plt.xlabel('Time')
# plt.show()


def draw_acf_pacf(ts, subtitle, lags=30):
    print("自相关图和偏自相关图,maxlags={}".format(lags))
    f = plt.figure(facecolor='white', figsize=(18,4))
    ax1 = f.add_subplot(121)
    plot_acf(ts, lags=lags, ax=ax1, title='ACF\n{}'.format(subtitle))
    ax2 = f.add_subplot(122)
    plot_pacf(ts, lags=lags, ax=ax2, title='PACF\n{}'.format(subtitle))

# 使用acf函数与pacf函数绘制差分后的数据的自相关图与偏自相关图
# draw_acf_pacf(call_2[1:], 'dotcom_calls - 52步差分后再做1阶差分', lags=30)
# plt.show()
# 自相关图和偏自相关图中不能明确地确定模型的阶数，所以使用AIC准则进行定阶
best_aic = np.inf # 表示一个无限大的正数
best_order = None
best_mdl = None
pq_rng = range(2)
d_rng = range(1)

for p in pq_rng:
    for d in d_rng:
        for q in pq_rng:
            try:
                tmp_mdl = smt.ARIMA(call_2[1:], order=(p, d, q)).fit()
                tmp_aic = tmp_mdl.aic
                print('aic : {:6.5f}| order: {}'.format(tmp_aic, (p, d, q)))
                if tmp_aic < best_aic:
                    best_aic = tmp_aic
                    best_order = (p, d, q)
                    best_mdl = tmp_mdl
            except:
                continue
print('\naic: {:6.5f}| order: {}'.format(best_aic, best_order))
# 通过AIC准则可以判断经过差分处理后的数据应使用ARIMA(1, 0, 1)模型

call_arima_a = smt.ARIMA(call_2[1:], order=(1, 0, 1)).fit()
print(call_arima_a.summary())

# 残差序列时序图 
# plt.plot(call_arima_a.resid)
# plt.ylabel('Residual')
# plt.show()

# 残差ACF图和PACF图
draw_acf_pacf(call_arima_a.resid, '残差', lags=30)
# plt.show()
# 从图中可以看出，残差已经无信息可提取。在残差自相关图中，残差滞后各期均无显著的自相关性。
# 在残差偏自相关图中，残差滞后各期也无显著的偏自相关性。

# 使用模型对测试数据进行预测
call_arima_final = smt.SARIMAX(train_call, order=(1, 1, 1), seasonal_order=(0, 1, 0, 52)).fit()
fig, ax = plt.subplots(figsize=(10, 8))
fig = plot_predict(call_arima_final, start="2014-07-13", end="2016-02-03", ax=ax)
legend = ax.legend(loc="upper left")
plt.show()