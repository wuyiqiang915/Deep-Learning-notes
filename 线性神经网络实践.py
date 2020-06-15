import numpy as np
import matplotlib.pyplot as plt

# 输入数据
X = np.array([[1, 0, 0, 0, 0, 0],  # 解决异或问题
              [1, 0, 1, 0, 0, 1],
              [1, 1, 0, 1, 0, 0],
              [1, 1, 1, 1, 1, 1]])

# 存为标签（一一对应数据）
Y = np.array([-1, 1, 1, -1])

# 随机权值，三行一列，取值范围（-1，1）
W = np.random.random(6) * 2 - 1

print('W=', W)
# 设置学习率
lr = 0.11

# 设置迭代次数
n = 0
# 设置输出值
O = 0


def update():
    global X, Y, W, lr, n
    n += 1
    O = np.dot(X, W.T)
    W_C = lr * ((Y - O.T).dot(X)) / X.shape[0]  # 平均权值
    W = W_C + W  # 修改权值


for _ in range(1000):
    update()  # 更新权值
    print(W)
    print(n)
    O = np.dot(X, W.T)  # 计算神经网络输出
    if (O == Y.T).all():  # 如果实际输出等于期望输出，模型收敛
        print("finished")
        print("epoch:", n)
        break

# 正样本（标签为1）
X1 = [1, 0]
Y1 = [0, 1]

# 负样本（标签为0）
X2 = [0, 1]
Y2 = [0, 1]


def calculate(x, root):
    a = W[5]
    b = W[2] + x * W[4]
    c = W[0] + x * W[1] + x * x * W[3]
    if root == 1:
        return (-b + np.sqrt(b * b - 4 * a * c)) / (2 * a)
    if root == 2:
        return (-b - np.sqrt(b * b - 4 * a * c)) / (2 * a)


# 作图
xdata = np.linspace(-1, 2)

plt.figure()

plt.plot(xdata, calculate(xdata, 1), 'r')
plt.plot(xdata, calculate(xdata, 2), 'r')

plt.plot(X1, Y1, 'bo')
plt.plot(X2, Y2, 'yo')
plt.show()
