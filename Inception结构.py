from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model
import graphviz
inputs = Input(shape=(28, 28, 192))
tower_1 = Conv2D(filters=64, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs)
tower_2 = Conv2D(filters=96, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs)
tower_2 = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu')(tower_2)
tower_3 = Conv2D(filters=16, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(inputs)
tower_3 = Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1), padding='same', activation='relu')(tower_3)
pooling = MaxPool2D(pool_size=(3, 3), strides=(1, 1), padding='same')(inputs)
pooling = Conv2D(filters=32, kernel_size=(1, 1), strides=(1, 1), padding='same', activation='relu')(pooling)
outputs = concatenate([tower_1, tower_2, tower_3, pooling], axis=3)
model = Model(inputs=inputs, outputs=outputs)
#model.summary()
plot_model(model=model, to_file='model.png', show_shapes=True, dpi=200)




