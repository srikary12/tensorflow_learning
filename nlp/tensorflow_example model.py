import keras.callbacks
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv('alzheimers_disease_data.csv')

# print(df)

data_df = df[['BMI', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality', 'Diagnosis']]

print(data_df)

def split(data_df):
    return data_df[['BMI', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality']], data_df[['Diagnosis']]

X,Y = split(data_df)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.1, shuffle=True)
# print(X_train.shape)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='sigmoid', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='sigmoid'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

class pltCallBack(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.losses = []
        self.accuracy = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.accuracy.append(logs.get('accuracy'))

    def on_train_end(self, logs=None):
        self.plot()

    def plot(self):
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.losses, label='Loss')
        plt.title('Loss over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.accuracy, label='Accuracy')
        plt.title('Accuracy over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.show()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, Y_train, epochs=100, callbacks=[pltCallBack()])

loss, accuracy = model.evaluate(X_test, Y_test)

print(loss, accuracy)