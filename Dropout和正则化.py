import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.losses import CategoricalCrossentropy
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.regularizers import L2

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255, x_test / 255
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

model1 = Sequential([Flatten(input_shape=(28, 28)),
                     Dense(units=200, activation='tanh', kernel_regularizer=L2(0.0003)),
                     Dense(units=100, activation='tanh', kernel_regularizer=L2(0.0003)),
                     Dense(units=10, activation='softmax', kernel_regularizer=L2(0.0003))
                     ])
model2 = Sequential([Flatten(input_shape=(28, 28)),
                     Dense(units=200, activation='tanh'),
                     Dense(units=100, activation='tanh'),
                     Dense(units=10, activation='softmax')
                     ])
sgd = SGD(0.2)
# 标签平滑化
loss1 = CategoricalCrossentropy(label_smoothing = 0.1)
loss2 = CategoricalCrossentropy(label_smoothing = 0.0)
model1.compile(optimizer=sgd,
               loss=loss1,
               metrics=['accuracy'])
model2.compile(optimizer=sgd,
               loss=loss2,
               metrics=['accuracy'])
epochs = 30
batch_size = 32

history1 = model1.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))
history2 = model2.fit(x_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test))

plt.plot(np.arange(epochs), history1.history['val_accuracy'], c='b', label='L2 Regularization')
# 画出model2验证集准确率曲线图
plt.plot(np.arange(epochs), history2.history['val_accuracy'], c='y', label='FC')
# 图例
plt.legend()
# x 坐标描述
plt.xlabel('epochs')
# y 坐标描述
plt.ylabel('accuracy')
# 显示图像
plt.show()
