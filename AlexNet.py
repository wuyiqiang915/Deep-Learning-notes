import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler

num_classes = 17
batch_size = 32
epochs = 100
image_size = 224

train_datagen = ImageDataGenerator(rotation_range=20,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   rescale=1 / 255,
                                   shear_range=10,
                                   zoom_range=0.1,
                                   horizontal_flip=True,
                                   brightness_range=(0.8, 1.3),
                                   fill_mode='nearest', )
test_datagen = ImageDataGenerator(rescale=1 / 255, )

train_generator = train_datagen.flow_from_directory('data/train',
                                                    target_size=(image_size, image_size),
                                                    batch_size=batch_size, )
test_generator = test_datagen.flow_from_directory('data/test',
                                                  target_size=(image_size, image_size),
                                                  batch_size=batch_size,)

print(train_generator.class_indices)

model = Sequential()
model.add(Conv2D(filters=96,
                 kernel_size=(11, 11),
                 strides=(4, 4),
                 padding='valid',
                 input_shape=(image_size, image_size, 3),
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3),
                       strides=(2, 2),
                       padding='valid'))
model.add(Conv2D(filters=256,
                 kernel_size=(5, 5),
                 strides=(1, 1),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3),
                       strides=(2, 2),
                       padding='valid'))
model.add(Conv2D(filters=384,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='same',
                 activation='relu'))
model.add(Conv2D(filters=384,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='same',
                 activation='relu'))
model.add(Conv2D(filters=256,
                 kernel_size=(3, 3),
                 strides=(1, 1),
                 padding='same',
                 activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3),
                       strides=(2, 2),
                       padding='valid'))
model.add(Flatten())
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4096, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
model.summary()


def adjust_learning_rate(epoch):
    if epoch <= 30:
        lr = 1e-4
    elif epoch > 30 and epoch <= 70:
        lr = 1e-5
    else:
        lr = 1e-6
    return lr


adam = Adam(lr=1e-4)
callbacks = []
callbacks.append(LearningRateScheduler(adjust_learning_rate))
model.compile(optimizer=adam,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(x=train_generator,
                    epochs=epochs,
                    validation_data=test_generator,
                    callbacks=callbacks)

# 画出训练集准确率曲线图
plt.plot(np.arange(epochs), history.history['accuracy'], c='b', label='train_accuracy')
# 画出验证集准确率曲线图
plt.plot(np.arange(epochs), history.history['val_accuracy'], c='y', label='val_accuracy')
# 图例
plt.legend()
# x 坐标描述
plt.xlabel('epochs')
# y 坐标描述
plt.ylabel('accuracy')
# 显示图像
plt.show()
