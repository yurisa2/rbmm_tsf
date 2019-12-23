import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


def standardize_col_roll(col, window, func='standard'):

    def standardize_rolling(x):
        return ((x - np.mean(x)) / np.std(x))[-1]

    def normalize_rolling(x):
        return ((x - np.min(x)) / (np.max(x) - np.min(x)))[-1]

    if func == 'standard':
        col_standardized = col.rolling(window).apply(standardize_rolling)
    else:
        col_standardized = col.rolling(window).apply(normalize_rolling)

    return col_standardized


colnames = ['DATE', 'O', 'H', 'L', 'C', 'V1', 'V2']
win = pd.read_csv("data/winm1.csv", names=colnames, header=None, encoding='utf-16')

closing_price = win["C"]
results_price = win["C"]

results_price = results_price.shift(-1)

closing_price = closing_price[:-1]
results_price = results_price[:-1]

closing_price = standardize_col_roll(closing_price, 20, func='norm')
results_price = standardize_col_roll(results_price, 20, func='norm')

closing_price = closing_price.dropna()
results_price = results_price.dropna()


model = Sequential()
model.add(Dense(100, activation='relu', input_dim=1))
model.add(Dense(1, activation='softmax'))
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy', 'categorical_accuracy']
              )
