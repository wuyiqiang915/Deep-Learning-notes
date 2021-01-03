import numpy as np
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

n = 3
target_name = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
plt.imshow(x_train[n])
plt.axis('off')
plt.title(target_name[y_train[n][0]])
plt.show()

x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

model = Sequential()
model.add(Convolution2D(input_shape=(32, 32, 3),
                        filters=32,
                        kernel_size=3,
                        strides=1,
                        padding='same',
                        activation='relu'))
model.add(Convolution2D(filters=32,
                        kernel_size=3,
                        strides=1,
                        padding='same',
                        activation='relu'))
model.add(MaxPooling2D(pool_size=2,
                       strides=2,
                       padding='valid'))
model.add(Dropout(0.2))
model.add(Convolution2D(filters=64,
                        kernel_size=3,
                        strides=1,
                        padding='same',
                        activation='relu'))
model.add(Convolution2D(filters=64,
                        kernel_size=3,
                        strides=1,
                        padding='same',
                        activation='relu'))
model.add(MaxPooling2D(pool_size=2,
                       strides=2,
                       padding='valid'))
model.add(Dropout(0.3))
model.add(Convolution2D(filters=128,
                        kernel_size=3,
                        strides=1,
                        padding='same',
                        activation='relu'))
model.add(Convolution2D(filters=128,
                        kernel_size=3,
                        strides=1,
                        padding='same',
                        activation='relu'))
model.add(MaxPooling2D(pool_size=2,
                       strides=2,
                       padding='valid'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

adam = Adam(lr=1e-4)
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics='accuracy')
model.fit(x_train, y_train,
          batch_size=64,
          epochs=100,
          validation_data=(x_test, y_test),
          shuffle=True)
