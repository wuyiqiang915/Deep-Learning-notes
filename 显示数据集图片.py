from sklearn.datasets import load_digits
import pylab as pl

digits = load_digits()  # 载入数据集
print(digits.data.shape)

pl.gray()   # 灰度化
pl.matshow(digits.images[0])
pl.show()