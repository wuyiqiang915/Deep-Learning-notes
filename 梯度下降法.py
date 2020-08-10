import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt("data.csv", delimiter=",")  # 通过np载入数据，因为在同一个目录下所以直接导入，delimiter=","表示以，号分割数据
x_data = data[:, 0]  # :号两边没有东西表示取全部数据，“，0”表示取第0列
y_data = data[:, 1]  # 同理取第一列
# plt.scatter(x_data, y_data)
# plt.show()  # 显示数据

# 学习率设置
lr = 0.0001
# 截距
b = 0
# 斜率
k = 0
# 最大迭代数
epochs = 50


# 最小二乘法
# 计算代价函数
def compute_error(b, k, x_data, y_data):
    totalError = 0
    for i in range(0, len(x_data)):
        totalError += (y_data[i] - (k * x_data[i] + b)) ** 2
    return totalError / float(len(x_data)) / 2.0


# 梯度下降法
def gradient_descent_runner(x_data, y_data, b, k, lr, epochs):
    # 计算总数据量
    m = float(len(x_data))
    # 循环迭代次数
    for i in range(epochs):
        b_grad = 0  # 设置变量存放临时数据
        k_grad = 0
        # 计算梯度的综合在求平均
        for j in range(0, len(x_data)):
            b_grad += (1 / m) * (((k * x_data[j]) + b) - y_data[j])
            k_grad += (1 / m) * (((k * x_data[j]) + b) - y_data[j]) * x_data[j]
        # 更新b和k
        b = b - (lr * b_grad)
        k = k - (lr * k_grad)

        # if i % 5 == 0:
        #     print("epochs:",i)
        #     plt.plot(x_data,y_data,'b.')
        #     plt.plot(x_data,k*x_data +b,'r')
        #     plt.show()

    return b, k


print("starting b ={0},k = {1}, error = {2}".format(b, k, compute_error(b, k, x_data, y_data)))
print("running...")
b, k = gradient_descent_runner(x_data, y_data, b, k, lr, epochs)
print("after(0) iterations b ={1},k = {2}, error = {3}".format(epochs, b, k, compute_error(b, k, x_data, y_data)))

plt.plot(x_data, y_data, 'b.')     # b. 是x，y以点的形式画出来
plt.plot(x_data, k * x_data + b, 'r')
plt.show()
