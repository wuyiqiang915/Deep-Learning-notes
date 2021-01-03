import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import plot_model
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPool2D, GlobalAvgPool2D, BatchNormalization, Activation, \
    Add, ZeroPadding2D
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import Model

# 类别数
num_classes = 17
# 批次大小
batch_size = 32
# 周期数
epochs = 100
# 图片大小
image_size = 224

# 训练集数据进行数据增强
train_datagen = ImageDataGenerator(
    rotation_range=20,  # 随机旋转度数
    width_shift_range=0.1,  # 随机水平平移
    height_shift_range=0.1,  # 随机竖直平移
    rescale=1 / 255,  # 数据归一化
    shear_range=10,  # 随机错切变换
    zoom_range=0.1,  # 随机放大
    horizontal_flip=True,  # 水平翻转
    brightness_range=(0.7, 1.3),  # 亮度变化
    fill_mode='nearest',  # 填充方式
)
# 测试集数据只需要归一化就可以
test_datagen = ImageDataGenerator(
    rescale=1 / 255,  # 数据归一化
)

# 训练集数据生成器，可以在训练时自动产生数据进行训练
# 从'data/train'获得训练集数据
# 获得数据后会把图片resize为image_size×image_size的大小
# generator每次会产生batch_size个数据
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(image_size, image_size),
    batch_size=batch_size,
)

# 测试集数据生成器
test_generator = test_datagen.flow_from_directory(
    'data/test',
    target_size=(image_size, image_size),
    batch_size=batch_size,
)

# 定义残差单元
def block(x, filters, strides=1, conv_shortcut=True):
    # projection shortcut
    if conv_shortcut == True:
        shortcut = Conv2D(filters * 4, kernel_size=1, strides=strides, padding='valid')(x)
        # epsilon为BN公式中防止分母为零的值
        shortcut = BatchNormalization(epsilon=1.001e-5)(shortcut)
    else:
        # identity_shortcut
        shortcut = x
    # 3个卷积层
    x = Conv2D(filters=filters, kernel_size=1, strides=strides, padding='valid')(x)
    x = BatchNormalization(epsilon=1.001e-5)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
    x = BatchNormalization(epsilon=1.001e-5)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters=filters * 4, kernel_size=1, strides=1, padding='valid')(x)
    x = BatchNormalization(epsilon=1.001e-5)(x)

    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x


# 堆叠残差单元
def stack(x, filters, blocks, strides):
    x = block(x, filters, strides=strides)
    for i in range(blocks - 1):
        x = block(x, filters, conv_shortcut=False)
    return x


# 定义ResNet50
inputs = Input(shape=(image_size, image_size, 3))
# 填充3圈0，填充后图像从224×224变成230×230
x = ZeroPadding2D((3, 3))(inputs)
x = Conv2D(filters=64, kernel_size=7, strides=2, padding='valid')(x)
x = BatchNormalization(epsilon=1.001e-5)(x)
x = Activation('relu')(x)
# 填充1圈0
x = ZeroPadding2D((1, 1))(x)
x = MaxPool2D(pool_size=3, strides=2, padding='valid')(x)
# 堆叠残差结构
# blocks表示堆叠数量
x = stack(x, filters=64, blocks=3, strides=1)
x = stack(x, filters=128, blocks=4, strides=2)
x = stack(x, filters=256, blocks=6, strides=2)
x = stack(x, filters=512, blocks=3, strides=2)
# 根据特征图大小进行平均池化，池化后得到2维数据
x = GlobalAvgPool2D()(x)
x = Dense(num_classes, activation='softmax')(x)
# 定义模型
model = Model(inputs=inputs, outputs=x)
plot_model(model=model, to_file='model.png', show_shapes=True, dpi=200)

# model.summary()

# 学习率调节函数，逐渐减小学习率
def adjust_learning_rate(epoch):
    # 前40周期
    if epoch <= 40:
        lr = 1e-4
    # 前40到80周期
    elif epoch > 40 and epoch <= 80:
        lr = 1e-5
    # 80到100周期
    else:
        lr = 1e-6
    return lr

# 定义优化器
adam = Adam(lr=1e-4)

# 定义学习率衰减策略
callbacks = []
callbacks.append(LearningRateScheduler(adjust_learning_rate))

# 定义优化器，loss function，训练过程中计算准确率
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

# Tensorflow2.1版本(包括2.1)之后可以直接使用fit训练模型
history = model.fit(x=train_generator, epochs=epochs, validation_data=test_generator, callbacks=callbacks)

# 画出训练集准确率曲线图
plt.plot(np.arange(epochs), history.history['accuracy'], c='b', label='train_accuracy')
# 画出验证集准确率曲线图
plt.plot(np.arange(epochs), history.history['val_accuracy'], c='y', label='val_accuracy')
# 图例
plt.legend()
# x坐标描述
plt.xlabel('epochs')
# y坐标描述
plt.ylabel('accuracy')
# 显示图像
plt.show()
# 模型保存
model.save('ResNet50.h5')

