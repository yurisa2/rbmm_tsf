import numpy as np
import pandas as pd

import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

import data_prep as dtp

colnames = ['DATE', 'O', 'H', 'L', 'C', 'V1', 'V2']
win = pd.read_csv("data/winm1.csv", names=colnames, header=None, encoding='utf-16')
wdo = pd.read_csv("data/wdom1.csv", names=colnames, header=None, encoding='utf-16')


dataset = dtp.build_dataset(win,wdo,30, 10)

x = dataset["C_x"]

x_array = np.array(x)


y = dataset["target"]

len(x_array)
len(y)


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_array, y, test_size=0.3)


y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

type(y_test)

model = Sequential()
model.add(Dense(100, activation='relu', input_dim=1))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam',
              loss='binary_crossentropy')

model.fit(X_train,
          y_train,
          validation_data=(X_test, y_test),
          epochs=200,
          verbose=0)
