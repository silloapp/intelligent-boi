import argparse
from csv import reader
import matplotlib.pyplot as plt
import numpy as np
import pickle
import re

from keras.preprocessing.text import one_hot
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers.core import Activation, Dropout, Dense
from keras.layers import Flatten
from keras.layers import GlobalMaxPooling1D
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.text import Tokenizer

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", type=str, default='sentiment.model',
    help="path to output model")
ap.add_argument("-d", "--dataset", type=str, default='dataset.csv',
    help="path to input dataset")
ap.add_argument("-p", "--plot", type=str, default="plot.png",
    help="path to output loss/accuracy plot")
args = vars(ap.parse_args())

# initialize the number of epochs to train for, initial learning rate,
# and batch size
EPOCHS = 30
INIT_LR = 1e-3
BS = 32

train_set = []

#download imdb dataset from https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews
with open(args["dataset"], 'r') as read_obj:
    csv_reader = reader(read_obj)
    train_set = list(csv_reader)

TAG_RE = re.compile(r'<[^>]+>')
def remove_tags(text):
    return TAG_RE.sub('', text)
def preprocess_text(sen):
    # Removing html tags
    sentence = remove_tags(sen)
    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)
    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    return sentence

X = []
y = []

#load imdb dataset
for data in train_set:
    X.append(preprocess_text(data[0]))
    y.append(1) if data[1] == "positive" else y.append(0)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# src for converting raw text to NLP embeddings: 
# https://stackabuse.com/python-for-nlp-movie-sentiment-analysis-using-deep-learning-in-keras/
tokenizer = Tokenizer(num_words=5000)
tokenizer.fit_on_texts(X_train)
X_train = tokenizer.texts_to_sequences(X_train)
X_test = tokenizer.texts_to_sequences(X_test)

# padding
vocab_size = len(tokenizer.word_index) + 1
maxlen = 200
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)

# create feature matrix (embeddings dictionary must be loaded in load_embeddings.py)
embeddings_dictionary = pickle.load(open( "embeddings.p", "rb" ))
embedding_matrix = np.zeros((vocab_size, 200))
for word, index in tokenizer.word_index.items():
    embedding_vector = embeddings_dictionary.get(word)
    if embedding_vector is not None:
        embedding_matrix[index] = embedding_vector

model = Sequential()
embedding_layer = Embedding(vocab_size, 200, weights=[embedding_matrix], input_length=maxlen , trainable=False)
model.add(embedding_layer)
model.add(LSTM(256))
model.add(Dense(1, activation='sigmoid'))
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])
print(model.summary())

history = model.fit(X_train, y_train, batch_size=BS, epochs=EPOCHS, verbose=1, validation_split=0.2)

score = model.evaluate(X_test, y_test, verbose=1)

print("Test Score:", score[0])
print("Test Accuracy:", score[1])

print("[INFO] serializing network...")
model.save(args["model"])

#plot training history
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])

plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])

plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'], loc='upper left')
plt.savefig(args["plot"])
