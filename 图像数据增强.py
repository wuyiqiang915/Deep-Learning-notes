from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
import numpy as np

datagen = ImageDataGenerator(rotation_range=40,  # 随机旋转度数
                             width_shift_range=0.2,  # 随机水平平移
                             height_shift_range=0.2,  # 随机竖直平移
                             rescale=1 / 255,  # 数据归一化
                             shear_range=30,  # 随机错切变换
                             zoom_range=0.2,  # 随机放大
                             horizontal_flip=True,  # 水平翻转
                             brightness_range=(0.7, 1.3),  # 亮度变化
                             fill_mode='nearest',  # 填充方式
                             )
# 载入图片
img = load_img('boo.jpg')
# 把图片变成array，此时数据是3 维
# 3 维(height,width,channel)
x = img_to_array(img)
# 在第0 个位置增加一个维度
# 我们需要把数据变成4 维，然后再做数据增强
# 4 维(1,height,width,channel)
x = np.expand_dims(x, 0)
# 生成20 张图片
i = 0
# 生成的图片都保存在boo 文件夹中，文件名前缀为new_boo,图片格式为jpeg
for batch in datagen.flow(x, batch_size=1, save_to_dir='boo', save_prefix='new_boo', save_format='jpeg'):
    i += 1
    print("running...")
    if i == 20:
        break
