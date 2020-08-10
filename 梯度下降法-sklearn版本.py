from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

# 导入数据，并且以“，”为分隔符分割数据
data = np.genfromtxt("data.csv", delimiter=",")
# x_data = data[:, 0]
# y_data = data[:, 1]

x_data = data[:, 0, np.newaxis]  # np.newaxis 给数据加上一个维度 例这里数据就从（100，）变成了（100,1）二维格式
y_data = data[:, 1, np.newaxis]

# 创建并拟合模型
model = LinearRegression()
model.fit(x_data, y_data)

# 绘图
plt.plot(x_data, y_data, 'b.')
plt.plot(x_data, model.predict(x_data), 'r')
plt.show()
