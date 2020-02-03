import numpy as np  # 基于矩阵的运算

# 把一个列表转化为矩阵
array = np.array([[1, 2, 3], [2, 3, 4]])
print(array)
# 几维数组
print('number of dim:', array.ndim)
# 几行几列
print('shape:', array.shape)
# size 总共有几个元素
print('size:', array.size)
# ----------numpy创建array-----------
# 定义数组里的数据类型
# np.int, np.float, np.float32
a = np.array([[2, 23, 4],
              [2, 3, 4]], dtype=np.int)
# 几行几列的0矩阵 一定要有括号
a = np.zeros((3, 4))
a = np.ones((3, 4), dtype=np.int16)
a = np.empty((3, 4))
# row vector[10 12 14 16 18] 从10-20 step size=2
a = np.arange(10, 20, 2)
# 0-11 default step size = 1, 0-n
a = np.arange(12).reshape((3, 4))
# 从1-10有20段
a = np.linspace(1, 10, 6).reshape((2, 3))

# --------------------numpy基础运算-------------------
# 对向量的操作
a = np.array([10, 20, 30, 40])
b = np.arange(4)
print(a, b)
c = a-b
# 对每个量求平方
c = b**2
# 对应量分别相乘 相当于点乘
c = a*b
# np.sin(), np.cos(), np.tan()
c = 10*np.sin(a)
# 大于小于 boolean
print(b < 3)
print(b)
# 对矩阵的操作
a = np.array([[1, 1], [0, 1]])
b = np.arange(4).reshape((2, 2))
# 逐个相乘
c = a*b
# 矩阵相乘 两种形式
c_dot = np.dot(a, b)
c_dot_2 = a.dot(b)

# 随机生成的矩阵 0-1的数字
a = np.random.random((2, 4))
summation = np.sum(a)
# 对行求和 a[:][j] 列是0，行是1
summationRow = np.sum(a, axis=1)
# 对列求和 a[i][:]
summationCol = np.sum(a, axis=0)
minimal = np.min(a)
maximal = np.max(a)

A = np.arange(14, 2, -1).reshape((3, 4))
# 找矩阵中最小值的index索引,从0开始
minIndex = np.argmin(A)
maxIndex = np.argmax(A)
# 均值
meanMatrix = np.mean(A)
meanMatrix = A.mean()
meanMatrix = np.average(A)
# 中位数
medianMatrix = np.median(A)
# 累加的值
cumSum = np.cumsum(A)
# 相邻两个数的差距 后-前
different = np.diff(A)
# 输出的非0数 输出两个数组 第几行第几列非0
nonZero = np.nonzero(A)
# 对A逐行排序 从小到大
sortRow = np.sort(A)
# 矩阵反向
trans = np.transpose(A)
trans = A.T
# A^T*A
A_square = A.T.dot(A)
# 把矩阵中小于a的数变成a(5),大于b的数变成b(9), 之间的数不变
A_change = np.clip(A, 5, 9)

# -----------------numpy的索引---------------------
A = np.arange(3, 15).reshape((3, 4))
print(A)
# 索引行
row2 = A[2]
# 索引值
value = A[1][1]
value = A[1, 1]
# 第二行的所有数
value = A[2, :]
# 第一列的所有数
value = A[:, 1]
# A[1][1] A[1][2] 右边都是开区间
value = A[1, 1:3]
# 对行迭代
for row in A:
    print(row)
# 对列迭代
for column in A.T:
    print(column)
# 对值迭代 flat is an iterator ;flatten 把矩阵拉直成向量
for item in A.flat:
    print(item)

# --------------numpy array合并--------------------
A = np.array([1, 1, 1])
B = np.array([2, 2, 2])
# vertical stack 上下合并
C = np.vstack((A, B))
# horizontal stack 左右合并
D = np.hstack((A, B))
# 把行向量变成列向量 在纵向加一个维度
E = A[:, np.newaxis]
print(A.shape, E.shape)
A = np.array([1, 1, 1])[:, np.newaxis]
B = np.array([2, 2, 2])[:, np.newaxis]
# horizontal stack 左右合并
D = np.hstack((A, B))
# 多个array的合并 axis=0 上下合并，axis=1，左右合并
C = np.concatenate((A, B, B, A), axis=0)
print(D)
# ------------------numpy array分割-------------------
A = np.arange(12).reshape((3, 4))
# 横向等量分割
B = np.split(A, 2, axis=1)
# 纵向向等量分割
B = np.split(A, 3, axis=0)
# 不等量分割
B = np.array_split(A, 3, axis=1)
# 纵向分割
B = np.vsplit(A, 3)
# 横向分割
B = np.hsplit(A, 2)
print(A)
print(B)
# ------------------numpy copy and deep copy------------
a = np.arange(4, dtype=float)
# 把b指向a所指的地方，a b都是指针
b = a
c = a
d = b
a[0] = 0.3
print(a)
print(b)
# 把a的值取出来 但不关联a --> deep copy
b = a.copy()
