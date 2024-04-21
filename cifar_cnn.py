import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import keras

from keras import layers


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

# print(x_train.shape)

x_train = x_train.astype("float32")/255.0
x_test = x_test.astype("float32")/255.0

inputs = keras.Input(shape=(32,32,3))
x = layers.Conv2D(32, 3, padding="valid", activation='relu')(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, activation='relu')(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3, activation='relu')(x)
x = layers.Flatten()(x)
x = layers.Dense(64, activation='relu')(x)
output = layers.Dense(10, activation='softmax')(x)
model = keras.Model(inputs=inputs, outputs=output)

# print(model.summary())

model.compile(optimizer="adam",
              loss=tf.keras.losses.sparse_categorical_crossentropy,
              metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=1)
model.evaluate(x_test, y_test, batch_size=64, verbose=2)
