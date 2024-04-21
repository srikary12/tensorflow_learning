import os
import tensorflow as tf
import tensorflow_datasets as tfds

print(tf.__version__)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


mnist_data = tfds.load("fashion_mnist")
for item in mnist_data:
    print(item)


mnist_train = tfds.load(name="fashion_mnist", split="train")
assert isinstance(mnist_train, tf.data.Dataset)
print(type(mnist_data))
print(type(mnist_train))

for item in mnist_train.take(1):
    print(item.keys())
    print(type(item))
    print(item['image'])
    print(item['label'])
