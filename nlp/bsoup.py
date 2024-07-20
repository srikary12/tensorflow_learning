import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from bs4 import BeautifulSoup

# tf.config.list_physical_devices('GPU')
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

sentences = [
    'Today is a sunny day',
    'Today is a rainy day',
    'Is it sunny today?',
    'I really enjoyed walking in the snow today'
]

test_data = [
    'Today is a snowy day',
    'Will it be rainy tomorrow?'
]

tokenizer = Tokenizer(num_words=100, oov_token="<OOV>")
sequences = tokenizer.fit_on_texts(sentences)
word_index = tokenizer.word_index
print(word_index)
test_seq = tokenizer.texts_to_sequences(sentences)
print(test_seq)
padded = pad_sequences(test_seq, padding="post", maxlen=6, truncating='post')
print(padded)
soup = BeautifulSoup(str(sentences))
sentence = soup.get_text()