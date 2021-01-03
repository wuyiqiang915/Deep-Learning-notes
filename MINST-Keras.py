import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import EarlyStopping

# 载入数据集
minst = tf.keras.datasets.mnist
# 载入数据，数据载入的时候就已经划分好训练集和测试集
# 训练集数据x_train 的数据形状为（60000，28，28）
# 训练集标签y_train 的数据形状为（60000）
# 测试集数据x_test的数据形状为（10000，28，28）
# 测试集标签y_test的数据形状为（10000）
(x_train, y_train), (x_test, y_test) = minst.load_data()
# 对训练集和测试集的数据进行归一化处理，有助于提升模型训练速度
x_train, x_test = x_train / 255, x_test / 255
# 把训练集和测试集的标签转为独热编码
y_train = tf.keras.utils.to_categorical(y_train, num_classes=10)
y_test = tf.keras.utils.to_categorical(y_test, num_classes=10)
# 模型定义
# 先用Flatten 把数据从3 维变成2 维，(60000,28,28)->(60000,784)
# 设置输入数据形状input_shape不需要包含数据的数量，（28,28）即可
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                                    tf.keras.layers.Dense(10, activation='softmax')])
# sgd 定义随机梯度下降法优化器
# loss='mse'定义均方差代价函数
# metrics=['accuracy']模型在训练的过程中同时计算准确率
sgd = SGD(0.1)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
# EarlyStopping是Callbacks的一种，callbacks用于指定在每个epoch 或batch 开始和结束的时候进行哪种特定操作
# monitor='val_accuracy',监控验证集准确率
# patience=5,连续5 个周期没有超过最高的val_accuracy值，则提前停止训练
# verbose=1，停止训练时提示early stopping
early_stooping = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1)
# 传入训练集数据和标签训练模型
# 周期大小为10（把所有训练集数据训练一次称为训练一个周期）
# 批次大小为32（每次训练模型传入32 个数据进行训练）
# validation_data设置验证集数据
model.fit(x_train, y_train, epochs=100, batch_size=32, validation_data=(x_test, y_test), callbacks=[early_stooping])
model.save('my_model/mnist.h5')