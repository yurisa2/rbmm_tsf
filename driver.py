import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
plt.style.use('dark_background')
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, LSTM, GlobalMaxPooling1D, SpatialDropout1D

data = pd.read_csv("winprep.csv", header=0)

data.shape

x = data

del(x['target'])
x = np.array(x)


x.shape

X_train = x
y_train = y


look_back = 100

nb_samples = X_train.shape[0] - look_back
y = np.array(data['target'])

x_train_reshaped = np.zeros((nb_samples, look_back, 7))
y_train_reshaped = np.zeros((nb_samples))


samples = x_train_reshaped.shape[0]

x_train_reshaped.shape
y_train_reshaped.shape


for i in range(nb_samples):
    y_position = i + look_back
    x_train_reshaped[i] = X_train[i:y_position]
    y_train_reshaped[i] = y_train[y_position]



y_train_one_hot = to_categorical(y_train_reshaped)


model = Sequential()

model.add(LSTM(100, activation='relu', input_shape=(100, 7)))
model.add(Dense(3, activation='softmax')) #since number of output classes is 4
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()
model.fit(x_train_reshaped,
          y_train_one_hot,
          validation_split=0.3,
          epochs=10,
          # batch_size=3
          )
