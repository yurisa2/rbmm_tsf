import numpy as np
import pandas as pd

import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.preprocessing.sequence import TimeseriesGenerator
from sklearn.model_selection import train_test_split
from keras.utils.data_utils import Sequence

colnames = ['DATE', 'O', 'H', 'L', 'C', 'V1', 'V2']
win = pd.read_csv("data/winm1.csv", names=colnames, header=None, encoding='utf-16')
# wdo = pd.read_csv("data/wdom1.csv", names=colnames, header=None, encoding='utf-16')
win = win.iloc[-10000:]



time = np.array(range(1, len(win_new)+1))

win_new = np.array(win["C"])


full = pd.DataFrame()
full["time"] = time
full["win_new"] = win_new

full = np.array(full)

n_input = 2
generator = TimeseriesGenerator(full, full, length=n_input)
# number of samples
print('Samples: %d' % len(generator))
# print each sample
for i in range(len(generator)):
	x, y = generator[i]
	print('%s => %s' % (x, y))


# X_train, X_test, y_train, y_test = train_test_split(x_array, y, test_size=0.3)
generator.get_config()

# define model
model = Sequential()
model.add(Dense(8, activation='relu', input_dim=n_input))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

hist = model.fit_generator(generator)
