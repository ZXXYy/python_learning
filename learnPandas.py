# pandas similar to dictionary, 每行每列可以有名字
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# -----------------pandas 基本介绍-----------------------
# pandas 的列表 tuple
s = pd.Series([1, 3, 6, np.nan, 44, 1])
print(s)
# pandas 的data frame 类似于矩阵
dates = pd.date_range('20200202', periods=6)
print(dates)
df = pd.DataFrame(np.random.random((6, 4)), index=dates)
# 行的名字是date 列的名字是a，b，c，d default 是1，2，3，4
df = pd.DataFrame(np.random.random((6, 4)), index=dates, columns=['a', 'b', 'c', 'd'])
print(df)
# A,B,C... 是列的名字，：后面是该列的值
df2 = pd.DataFrame({'A': 1.,
                    'B': pd.Timestamp('20130102'),
                    'C': pd.Series(1, index=list(range(4)), dtype='float32'),
                    'D': np.array([3]*4, dtype='int32'),
                    'E': pd.Categorical(["test", "train", "test", "train"]),
                    'F': 'foo'})
# 查看每一列值的类型
print(df2.dtypes)
# 查看行的index值
print(df2.index)
# 查看列的column值
print(df2.columns)
# 查看每个值
print(df2.values)
# 运算numerical attribute的均值，方差等
print(df2.describe())
print(df2)
# 行列转换
print(df2.T)
# 对列的attribute排序，升序， ABCDEF
print(df2.sort_index(axis=1, ascending=False))
print(df2.sort_index(axis=0, ascending=False))
print(df2.sort_values(by='E'))

# ------------pandas 选择数据---------------------------

dates = pd.date_range('20200202', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['a', 'b', 'c', 'd'])
print(df)
# 对frame进行索引
temp = df['a'], df.a
# 选择第0-2行数据
temp = df[0:3]
temp = df['20200202':'20200204']
# =======select by label: loc(local)=========
temp = df.loc['20200202']
# 保留所有行，以及选择的列
temp = df.loc[:, ['a', 'b']]
# 选择某几行，某几列
temp = df.loc['20200202', ['a', 'b']]
# =======select by position: iloc 与numpy相似============
# 选择第三行的数据
temp = df.iloc[3]
# 选择出某个sub data frame
temp = df.iloc[3:5, 1:3]
temp = df.iloc[[1, 3, 5], 1:3]
# ====== mixed selection: ix=============== has been removed right now
# temp = df.ix[:3, ['A', 'C']]

# boolean indexing
# 在A中比8大的 来筛选tuple
temp = df[df.a > 8]
print(temp)

# ----------------pandas 设置值-------------------
dates = pd.date_range('20200202', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['a', 'b', 'c', 'd'])
df.iloc[2, 2] = 1111
df.loc['20200202', 'b'] = 2222
# 对整行全部赋值为0
# df[df.a > 4] = 0
# 仅对a进行赋值0
df.a[df.a > 4] = 0
# 添加一列
df['F'] = np.nan
df['E'] = pd.Series([1, 2, 3, 4, 5, 6], index=dates)

# --------------pandas处理丢失数据-----------------
dates = pd.date_range('20200202', periods=6)
df = pd.DataFrame(np.arange(24).reshape((6, 4)), index=dates, columns=['a', 'b', 'c', 'd'])
df.iloc[0, 1] = np.nan
df.iloc[1, 2] = np.nan
# 如果行里面有NaN 则整行丢掉 axis=0, axis=1(丢掉列)
# how ={'any','all'} any= 出现一个就全部丢掉，all = 全部NaN才丢掉
df2 = df.dropna(axis=0, how='any')
# 对确实的数据进行填充 value 为填入的值
df2 = df.fillna(value=0)
# 有没有NaN, true:缺失
df2 = df.isnull()
# 有一个等于true就为真
temp = (np.any(df.isnull()) == True)

# ----------pandas 导入导出数据-------------------
# read_csv, to_csv, read_json, to_json
data = pd.read_csv('student.csv')
# print(data)
# data.to_pickle('student.json')

# ----------pandas合并concat-----------------------
# concatenating
df1 = pd.DataFrame(np.ones((3, 4))*0, columns=['a', 'b', 'c', 'd'])
df2 = pd.DataFrame(np.ones((3, 4))*1, columns=['a', 'b', 'c', 'd'])
df3 = pd.DataFrame(np.ones((3, 4))*3, columns=['a', 'b', 'c', 'd'])
# 上下合并 0是竖向，1是横向,忽略原来的index
res = pd.concat([df1, df2, df3], axis=0, ignore_index=True)

# join,两种形式['inner','outer']
df1 = pd.DataFrame(np.ones((3, 4))*0, columns=['a', 'b', 'c', 'd'], index=[1, 2, 3])
df2 = pd.DataFrame(np.ones((3, 4))*1, columns=['b', 'c', 'd', 'e'], index=[2, 3, 4])
# 没有的地方全部用NaN填充 相当与outer join
res = pd.concat([df1, df2])
# 仅仅保留相同的部分
res = pd.concat([df1, df2], join='inner', ignore_index=True)

# join_axes
# 根据df1的index来填充，df2里面没有的NaN填充
# res = pd.concat([df1, df2], axis=1, join_axes=[df1.index])

# append
df1 = pd.DataFrame(np.ones((3, 4))*0, columns=['a', 'b', 'c', 'd'])
df2 = pd.DataFrame(np.ones((3, 4))*1, columns=['a', 'b', 'c', 'd'])
df3 = pd.DataFrame(np.ones((3, 4))*2, columns=['a', 'b', 'c', 'd'])
res = df1.append([df2, df3], ignore_index=True)
s1 = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
res = df1.append(s1, ignore_index=True)
print(res)

# ----------------pandas合并 merge-----------------------------

# merging two df by key/keys (may be used in database)
left = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})
res = pd.merge(left, right, on='key')
# consider two keys
left = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                     'key2': ['K0', 'K1', 'K0', 'K1'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})
right = pd.DataFrame({'key1': ['K0', 'K1', 'K1', 'K2'],
                      'key2': ['K0', 'K0', 'K0', 'K0'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})
# how=inner 仅考虑相同的key
# how={inner, outer, left, right}
res = pd.merge(left, right, on=['key1', 'key2'], how='inner')
# indicator
df1 = pd.DataFrame({'col': [0, 1],
                    'col_left': ['a', 'b']})
df2 = pd.DataFrame({'col': [1, 2, 2],
                    'col_right': [2, 2, 2]})
res = pd.merge(df1, df2, on='col', how='outer', indicator=True)
# give the indicator a custom name
res = pd.merge(df1, df2, on='col', how='outer', indicator='indicator_column')

# merge by index
left = pd.DataFrame({'A': ['A0', 'A1', 'A2'],
                     'B': ['B0', 'B1', 'B2']},
                    index=['K0', 'K1', 'K2'])

right = pd.DataFrame({'C': ['C0', 'C1', 'C2'],
                     'D': ['D0', 'D1', 'D2']},
                     index=['K0', 'K2', 'K3'])
res = pd.merge(left, right, left_index=True, right_index=True, how='outer')
# how to deal with overlapping
boys = pd.DataFrame({'k': ['K0', 'K1', 'K2'], 'age': [1, 2, 3]})
girls = pd.DataFrame({'k': ['K0', 'K0', 'K3'], 'age': [4, 5, 6]})
res = pd.merge(boys, girls, on='k', how='inner', suffixes=['_boy', '_girl'])
# join to search

# ----------------------pandas plot -------------------------------------

# Series
data = pd.Series(np.random.randn(1000), index=np.arange(1000))
data = data.cumsum()
# data.plot()
# plt.show()

# DataFrame
data = pd.DataFrame(np.random.randn(1000, 4), index=np.arange(1000), columns=list("ABCD"))
data = data.cumsum()
print(data.head())
# color width parameter
# plot methods:
# 'bar', 'hist', 'box', 'kde', 'area', 'scatter', 'hexbin', 'pie'
# A column data-B column data
ax = data.plot.scatter(x='A', y='B', color='DarkBlue', label='Class1')
data.plot.scatter(x='A', y='C', color='DarkGreen', label='Class2', ax=ax)
# data.plot()
plt.show()

