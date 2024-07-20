import os
import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)


(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# print(x_train.shape)
x_train = x_train.reshape(-1, 28 * 28).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28 * 28).astype("float32") / 255.0


inputs = tf.keras.Input(shape={28 * 28})
# x = tf.keras.layers.Dense(512, activation=tf.nn.relu, name="first_layer")(inputs)
x = tf.keras.layers.Dense(128, activation=tf.nn.relu, name="second_layer")(inputs)
output = tf.keras.layers.Dense(10, activation=tf.nn.softmax)(x)
model = tf.keras.Model(inputs=inputs, outputs=output)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=["accuracy"],
)


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get('accuracy') > 0.85:
            print("\nReached 85% accuracy ending training at epoch ", epoch)
            self.model.stop_training = True


# callbacks = MyCallback()

model.fit(x_train, y_train, epochs=15, callbacks=[MyCallback()], batch_size=64)

model.evaluate(x_test, y_test)

classifications = model.predict(x_test[0].reshape(-1, 28 * 28))
print(classifications)

model.summary()
