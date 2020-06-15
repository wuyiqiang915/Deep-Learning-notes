import numpy as np

X = np.array([[1, 0, 0],  # 解决异或问题
              [1, 0, 1],
              [1, 1, 0],
              [1, 1, 1]])

# 存为标签（一一对应数据）
Y = np.array([[0, 1, 1, 0]])

# 随机权值，三行一列，取值范围（-1，1）
V = np.random.random((3, 4)) * 2 - 1
W = np.random.random((4, 1)) * 2 - 1
print('V=', V)
print('W=', W)
# 设置学习率
lr = 0.11

# 设置迭代次数
n = 0
# 设置输出值
O = 0


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def dsigmoid(x):
    return x * (1 - x)


def update():
    global X, Y, W, V, lr

    L1 = sigmoid(np.dot(X, V))  # 隐藏层输出（4，4）
    L2 = sigmoid(np.dot(L1, W))  # 输出层输出（4，1）

    L2_delta = (Y.T - L2) * dsigmoid(L2)
    L1_delta = L2_delta.dot(W.T) * dsigmoid(L1)

    W_C = lr * L1.T.dot(L2_delta)
    V_C = lr * X.T.dot(L1_delta)

    W = W_C + W
    V = V_C + V


for i in range(20000):
    update()  # 更新权值
    if i % 500 == 0:
        L1 = sigmoid(np.dot(X, V))
        L2 = sigmoid(np.dot(L1, W))
        print("Error:", np.mean(np.abs(Y.T - L2)))

L1 = sigmoid(np.dot(X, V))
L2 = sigmoid(np.dot(L1, W))
print(L2)


def judge(x):
    if x >= 0.5:
        return 1
    else:
        return 0


for i in map(judge, L2):
    print(i)
