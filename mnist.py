import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf


# print(tf.__version__)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(-1, 28*28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28*28).astype("float32") / 255.0

inputs = tf.keras.Input(shape=(28*28))
x = tf.keras.layers.Dense(512, activation=tf.keras.activations.hard_sigmoid, name = "firstLayer")(inputs)
y = tf.keras.layers.Dense(256, activation=tf.nn.sigmoid, name="secondLayer")(x)
outputs = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(y)
model = tf.keras.Model(inputs = inputs, outputs = outputs)

model.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=["accuracy"],
)

model.fit(x_train, y_train, batch_size=32, epochs=50, verbose=2)
evaluation = model.evaluate(x_test, y_test, batch_size=32, verbose=2)
print("Test Loss:", evaluation[0]*100)
print("Test Accuracy:", evaluation[1]*100)
# print(evaluation)
