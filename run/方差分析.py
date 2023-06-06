from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import pandas as pd
import sys

path = str(sys.path[0]).replace('\\run','')

data = pd.read_csv(path + '/data/creditcard_exp.csv')
df = data[['Income', 'edu_class']].dropna(how='any', axis=0)
df['edu_class'] = df['edu_class'].astype('str')
df['Income'] = df['Income'].astype('float')
formula = f'Income~C(edu_class)'

anova_results = anova_lm(ols(formula=formula, data=df).fit())
print(anova_results)

# 多因素方差分析 无交互效应
data_d = pd.read_csv(path + '/data/creditcard_exp.csv', encoding='utf-8-sig')
df_d = data_d[['avg_exp', 'edu_class', 'gender']].dropna(how='any', axis=0)
df_d['edu_class'] = df_d['edu_class'].astype('str')
df_d['gender'] = df_d['gender'].astype('str')
df_d['avg_exp'] = df_d['avg_exp'].apply(lambda x:0 if x==' ' else x)
df_d['avg_exp'] = df_d['avg_exp'].astype('float')

formula_d = f'avg_exp ~ C(edu_class) + C(gender)'
anvoa_d = ols(formula=formula_d, data=df_d).fit()
anova_results_d = anvoa_d.summary()
print(anova_results_d)

# 多因素方差分析 无交互效应
formula_d_y = f'avg_exp ~ C(edu_class) + C(gender) + C(edu_class) * C(gender)'
anvoa_d_y = ols(formula=formula_d_y, data=df_d).fit()
anova_results_d_y = anvoa_d_y.summary()
print(anova_results_d_y)

