import tensorflow as tf
from tensorflow.keras.models import load_model

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255, x_test / 255
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)

model = load_model('my_model/mnist.h5')
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))
