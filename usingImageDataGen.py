import urllib.request
# import zipfile
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3

# import keras


# url = "https://storage.googleapis.com/learning-datasets/horse-or-human.zip"
# validation_url = "https://storage.googleapis.com/learning-datasets/validation-horse-or-human.zip"

file_name = "horseorhuman.zip"
validation_file_name = "validationhorseorhuman.zip"

training_dir = "horse-or-human/training"
validation_dir = "horse-or-human/validation"
#
#
# urllib.request.urlretrieve(url, file_name)
# urllib.request.urlretrieve(validation_url, validation_file_name)
#
# zip_ref = zipfile.ZipFile(file_name, 'r')
# zip_ref.extractall(training_dir)
# zip_ref.close()
#
# zip_ref = zipfile.ZipFile(validation_file_name, 'r')
# zip_ref.extractall(validation_dir)
# zip_ref.close()


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

train_datagen = ImageDataGenerator(
    rescale=1 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
validation_datagen = ImageDataGenerator(rescale=1 / 255)

train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=(300, 300),
    class_mode='binary'
)
validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(300, 300),
    class_mode='binary'
)

weights_url = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
weights_file = "inception_v3.h5"
urllib.request.urlretrieve(weights_url, weights_file)
pre_trained_model = InceptionV3(input_shape=(150, 150, 3),
                                include_top=False,
                                weights=None)
pre_trained_model.load_weights(weights_file)
pre_trained_model.summary()

for layer in pre_trained_model.layers:
    layer.trainable = False

last_layer = pre_trained_model.get_layer('mixed7')
print('last layer output shape: ', last_layer.output_shape)
last_output = last_layer.output


# x = tf.keras.layers.Flatten()(last_output)
# # Add a fully connected layer with 1,024 hidden units and ReLU activation
# x = tf.keras.layers.Dense(1024, activation='relu')(x)
# # Add a dropout rate of 0.2
# x = tf.keras.layers.Dropout(0.2)(x)
# # Add a final sigmoid layer for classification
# x = tf.keras.layers.Dense(1, activation='sigmoid')(x)
#
# model = tf.keras.Model(pre_trained_model.input, x)
#
# model.compile(optimizer="rmsprop(lr=0.0001)",
#               loss='binary_crossentropy',
#               metrics=['acc'])

data_input = tf.keras.Input(shape=(300, 300, 3))
x = tf.keras.layers.Conv2D(16, (3, 3), activation='relu')(data_input)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
# x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(512, activation='relu')(x)
x = tf.keras.layers.Dense(128, activation='relu')(x)
output = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=data_input, outputs=output)

model.summary()

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=["accuracy"],
)


class MyCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs.get('val_accuracy') - logs.get('accuracy') < 0.05 and logs.get('accuracy') > 0.95:
            print("\nReached 95% accuracy ending training at epoch ", epoch)
            self.model.stop_training = True


my_callback = MyCallback()

model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator,
    callbacks=[my_callback],
)
