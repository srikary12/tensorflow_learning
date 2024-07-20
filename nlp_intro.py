import tensorflow as tf
from tensorflow import keras
# from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
import tensorflow_text as tf_text

sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny today?',
    'I really enjoyed walking in the snow today'
]

tokenizer = tf_text.Tokenizer()
tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)

test_data = [
    'Today is a snowy day',
    'Will it be rainy tomorrow?'
]

test_sequences = tokenizer.texts_to_sequences(sentences)
print(test_sequences)


padded = pad_sequences(test_sequences, maxlen=6)

print(padded)
