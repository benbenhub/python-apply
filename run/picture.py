import pandas as pd
import sys

path = str(sys.path[0]).replace('\\run','')

print('CSV的文件路径。')
print(pd.read_csv(path + '/data/sample.csv'))

print('sep分隔符默认为逗号')
print(pd.read_csv(path + '/data/sample.csv', sep = ','))

print('int或 list of ints 类型，0代表第一行，为列名，如果设定为None，则使用数值列名。')
print(pd.read_csv(path + '/data/sample.csv', header = 0))

print('list类型，重新定义列名，默认为None。')
print(pd.read_csv(path + '/data/sample.csv', names = ['id', 'code', 'name', 'role']))

print('list类型，读取指定列，设定后将缩短读取数据的时间与内存消耗，适合大数据量读取，默认为None。')
print(pd.read_csv(path + '/data/sample.csv', header = None, usecols = [2]))

print('dict类型，定义读取列的数据类型，默认为None。')
print(pd.read_csv(path + '/data/sample.csv', dtype = {'id':int, 'code':int, 'name':str,'role':str}))

print('int类型，读取大数据量的前多少行，默认为None。')
print(pd.read_csv(path + '/data/sample.csv', nrows = 1))

print('str、list或dict类型，指定读取为缺失值的值。')
print(pd.read_csv(path + '/data/sample.csv', na_values = ['333','444']))

print('bool类型，自动发现数据中的缺失值功能，默认打开为True，若确定数据无缺失，则可以设定为False。')
print(pd.read_csv(path + '/data/sample.csv', na_filter = True))

print('int类型，分块读取，当数据量较大时可以设定分块读取的行数，默认为None。若设定行数，则返回一个迭代器。')
print(pd.read_csv(path + '/data/sample.csv', chunksize = 1000))

print('str类型，数据的编码，Python3默认为utf-8，Python2默认为ASCII。')
print(pd.read_csv(path + '/data/sample.csv', encoding = 'utf-8'))

