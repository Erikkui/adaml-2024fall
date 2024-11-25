# %% [markdown]
# #  BM20A6100 Advanced Data Analysis and Machine Learning
# ## Erik Kuitunen, 0537275

# %% [markdown]
# Import packages

# %%
# Import packages
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import nltk
import tensorflow as tf

from keras.layers import Dense, Activation
from keras.layers import SimpleRNN
from keras.models import Sequential

from wordcloud import WordCloud
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer

# Check if GPU is available and set it as the default device
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
    print("GPU is available and will be used.")
else:
    print("GPU is not available, using CPU.")
    
print( physical_devices )

# %% [markdown]
# Read and preprocess text data

# %%
file = open("f:/Opiskelu/adaml-2024fall/week-10/robinhood.txt", 'rb')
lines = []
for line in file:
    line = line.strip().lower()
    line = line.decode("ascii", "ignore")
    
    if len(line) == 0:
        continue
    lines.append(line)
    
file.close()

text = " ".join(lines)
words = text.split()

# set of characters that occur in the text
chars = set( [c for c in text] )

# Total items in our vocabulary
unique_chars = len( chars )

# lookup tables to deal with indexes of characters rather than the characters themselves.
char2index = dict( (c, i) for i, c in enumerate( chars ) )
index2char = dict( (i, c) for i, c in enumerate( chars ) )

# %% [markdown]
# Reshaping and one-hot encoding of the data

# %%
sequence_length = 10
step = 1
input_chars = []
label_chars = []
for i in range( 0, len(text) - sequence_length, step ):
    input_chars.append(text[i:i + sequence_length])
    label_chars.append(text[i + sequence_length])
    
X = np.zeros((len(input_chars), sequence_length, unique_chars), dtype=bool)
y = np.zeros((len(input_chars), unique_chars), dtype=bool)
for i, input_char in enumerate(input_chars):
    for j, ch in enumerate(input_char):
        X[i, j, char2index[ch]] = 1
    y[i, char2index[label_chars[i]]] = 1
        

# %%
X.shape, i, j, ch, char2index[ch]

# %% [markdown]
# Model definition

# %%
HIDDEN_SIZE = 128
BATCH_SIZE = 128
NUM_ITERATIONS = 100
NUM_EPOCHS_PER_ITERATION = 1
NUM_PREDS_PER_EPOCH = 50

model = Sequential()
model.add(SimpleRNN(HIDDEN_SIZE, return_sequences=False, 
                    input_shape=(sequence_length, unique_chars), 
                    unroll=True))
model.add(Dense(unique_chars))
model.add(Activation("softmax"))

model.compile(loss="categorical_crossentropy", optimizer="rmsprop")


# %% [markdown]
# Training the model

# %%
for iteration in range(NUM_ITERATIONS):
    print("Iteration #: %d" % (iteration))
    
    model.fit(X, y, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS_PER_ITERATION)

# %% [markdown]
# Testing

# %%
test_idx = np.random.randint(len(input_chars))
test_chars = input_chars[test_idx]
predicted_text = test_chars

print("\nGenerating from seed: %s" % (test_chars))
# print(test_chars, end="")

for i in range(70):
    Xtest = np.zeros( (1, sequence_length, unique_chars) )
    for i, ch in enumerate(test_chars):
        Xtest[0, i, char2index[ch]] = 1
    pred = model.predict(Xtest, verbose=0)[0]
    ypred = index2char[np.argmax(pred)]
    predicted_text += ypred
    # print(ypred, end="\n")
    # move forward with test_chars + ypred
    test_chars = test_chars[1:] + ypred

# %%
predicted_text
