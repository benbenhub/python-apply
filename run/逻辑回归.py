import os
import numpy as np    
from scipy import stats
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import sys

path = str(sys.path[0]).replace('\\run','')

churn = pd.read_csv(path + '/data/telecom_churn.csv')
# print(churn.head())
# print(churn.info())
# 数据没有缺失情况

# 相关关系分析 
# 列联表
cross_table = pd.crosstab(churn.posTrend, churn.churn, margins=True)
# print(cross_table)
def percConvert(ser):
    return ser/float(ser[-1])
cross_table.apply(percConvert, axis=1)
# 列联表的卡方独立性检验：若显著，则说明二者存在相关关系
chi2, p, dof, expected_freq = stats.chi2_contingency(cross_table.iloc[:2, :2])
print(f"chisq = {chi2:.4f}\np-value = {p:.4f}\ndof = {dof}\nexpected_frep = {expected_freq}")
# 随机抽样 建立训练集与测试集
train = churn.sample(frac=0.7, random_state=666)
test = churn[~ churn.index.isin(train.index)]

print(' 训练集样本量: %i \n 测试样本量: %i' %(len(train), len(test)))

# 建立一元逻辑回归模型
lg = smf.logit('churn ~ duration', train).fit()
# 使用summary函数查看模型的一些信息
print(lg.summary())

# 变量选择 向前回归法 
def forward_select(data, response):
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = float('inf'), float('inf') 
    while remaining:
        aic_with_cadidates = []
        for candidate in remaining:
            formula = "{} ~ {}".format(response, ' + '.join(selected + [candidate]))
            aic = smf.logit(formula=formula, data=data).fit().aic
            aic_with_cadidates.append((aic, candidate))

        aic_with_cadidates.sort(reverse=True)
        best_new_score, best_candidate = aic_with_cadidates.pop()
        if current_score > best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
            print('aic is {},continuing!'.format(current_score))
        else:
            print('forward selection over!')
            break

        formula = "{} ~ {} ".format(response,' + '.join(selected))
        print('final formula is {}'.format(formula))
        model = smf.logit(formula=formula, data=data).fit()
        return model
# 对连续型变量进行筛选
candidates = ['churn', 'duration', 'AGE', 'incomeCode', 'nrProm', 'peakMinAv', 'peakMinDiff', 'call_10086']
data_for_select = train[candidates]
lg_ml = forward_select(data=data_for_select, response='churn')
lg_ml.summary()
print(f'原来有 {len(candidates)-1} 个变量')
print(f'筛选剩下 {len(lg_ml.params.index.tolist())} 个（包含 intercept 截距项）。')

# 对分类型变量进行筛选 判断个分类变量的显著性
class_col = ['posTrend', 'prom', 'edu_class', 'feton', 'curPlan', 'avgplan', 'gender', 'negTrend', 'planChange', 'posPlanChange', 'negPlanChange']
for i in class_col:
    tab = pd.crosstab(churn[i], churn.churn)
    print(i,''' p-value = %6.4f''' %stats.chi2_contingency(tab)[1])

# 建立逻辑回归模型
formula = '''churn ~ duration + AGE + incomeCode + peakMinDiff + call_10086 + C(edu_class) + C(feton) + C(posTrend) + 
C(negTrend) + C(curPlan) + C(avgplan) + C(gender)'''
lg_m = smf.logit(formula=formula, data=train).fit()
print(lg_m.summary())

# 方差膨胀因子检测
def vif(df, col_i):
    from statsmodels.formula.api import ols
    cols = list(df.columns)
    cols.remove(col_i)
    cols_noti = cols
    formula = col_i + '~' + '+'.join(cols_noti)
    r2 = ols(formula, df).fit().rsquared
    return 1. / (1. - r2)

exog = train[candidates].drop(['churn'], axis=1)
for i in exog.columns:
    print(i, '\t', vif(df=exog, col_i=i))

train['proba'] = lg_m.predict(train)
test['proba'] = lg_m.predict(test)
print(test.info())

# ROC曲线
# 模型的准确率
# acc_m = sum(test['prediction'] == test['churn']) / np.float(len(test))
# print('The accurancy is %.2f' %acc_m)

# 混淆矩阵
test['prediction'] = (test['proba'] > 0.5).astype('int')
print(pd.crosstab(test.churn, test.prediction, margins=True))

# ROC曲线
for i in np.arange(0.1, 0.9, 0.1):
    prediction = (test['proba'] > i).astype('int')
    confusion_matrix = pd.crosstab(prediction, test.churn, margins=True)
    precision = confusion_matrix.loc[0, 0] / confusion_matrix.loc['All', 0]
    recall = confusion_matrix.loc[0, 0] / confusion_matrix.loc[0, 'All']
    Specificity = confusion_matrix.loc[1, 1] / confusion_matrix.loc[1, 'All']
    f1_score = 2 * (precision * recall) / (precision + recall)
    print('threshold:%s, precision:%.2f, recall:%.2f, Specificity:%.2f, f1_score:%.2f' %(i, precision, recall, Specificity, f1_score))

import sklearn.metrics as metrics
fpr_test, tpr_test, th_test = metrics.roc_curve(test.churn, test.proba)
fpr_train, tpr_train, th_train = metrics.roc_curve(train.churn, train.proba)

print('AUC = %.4f' %metrics.auc(fpr_test, tpr_test))
# 计算最优阈值
print(th_test[(tpr_test - fpr_test).argmax()])

plt.figure(figsize=[3, 3])
plt.plot(fpr_test, tpr_test, 'b--')
plt.plot(fpr_train, tpr_train, 'r-')
plt.title('ROC curve')
plt.show()