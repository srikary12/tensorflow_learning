import tensorflow_datasets as tfds
import tensorflow as tf

imdb_sentences = []
train_data = tfds.as_numpy(tfds.load('imdb_reviews', split='train'))

for item in train_data:
    imdb_sentences.append(str(item['text']))

tokenizer = tf.keras.preprocessing.text.Tokenizer
tokenizer = tokenizer(num_words=5000)
tokenizer.fit_on_texts(imdb_sentences)

sequences = tokenizer.texts_to_sequences(imdb_sentences)

print(tokenizer.word_index)

from bs4 import BeautifulSoup
import string

stopwords = ["a", ... , "yourselves"]
table = str.maketrans('', '', string.punctuation)
