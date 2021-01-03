import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.optimizers import SGD

x_data = np.linspace(-0.5, 0.5, 200)[:, np.newaxis]
noise = np.random.normal(0, 0.02, x_data.shape)
y_data = np.square(x_data) + noise
plt.scatter(x_data, y_data)
plt.show()

model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(units=10, input_dim=1, activation='tanh'))
model.add(tf.keras.layers.Dense(units=1, activation='tanh'))
model.compile(optimizer=SGD(0.11), loss='mse')

for step in range(8001):
    cost = model.train_on_batch(x_data, y_data)
    if step % 1000 == 0:
        print("running...")
        plt.subplot(3, 3, step / 1000 + 1)
        prediction_value = model.predict(x_data)
        plt.scatter(x_data, y_data)
        plt.plot(x_data, prediction_value, 'r-', lw=2)
        plt.axis('on')
        plt.title("picture:" + str(int(step / 1000 + 1)))
plt.show()
