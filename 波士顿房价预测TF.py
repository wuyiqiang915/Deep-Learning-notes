import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import boston_housing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import plot_model

(train_data, train_targets), (test_data, test_target) = boston_housing.load_data()

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std
epochs = 200
model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(train_data.shape[1],)))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
history = model.fit(train_data, train_targets,
                    batch_size=32,
                    epochs=epochs,
                    validation_data=(test_data, test_target),
                    shuffle=True)

plot_model(model=model, to_file='model.png', show_shapes=True, dpi=200)

plt.plot(np.arange(epochs), history.history['mae'], c='b', label='train_mae')
# 画出验证集准确率曲线图
plt.plot(np.arange(epochs), history.history['val_mae'], c='y', label='val_mae')
# 图例
plt.legend()
# x坐标描述
plt.xlabel('epochs')
# y坐标描述
plt.ylabel('mae')
# 显示图像
plt.show()
