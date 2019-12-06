import pandas as pd
import numpy as np

import os

from matplotlib import pyplot as plt

import datetime

file = 'C:/Bitnami/wampstack-7.1.28-0/apache2/htdocs/LogicaFuzzy2019-2/misc/winm30.csv'

winm1 = pd.read_csv(file)

winm1.head()

winm1.describe()

winm1.iloc[:,0]

type(winm1.iloc[2,0])

winm1.iloc[:,0] = winm1.iloc[:,0].astype('datetime64[ns]')

type(winm1.iloc[2,0])

plt.plot(winm1.iloc[:,1])

winm1.columns = ["timestamp","open","high","low","close","vol1","vol2"]

winm1.describe()
winm1.head()

winm1.iloc[:,0].tail(200000)

winm1.iloc[:,0].size

winm1.count

# https://www.kaggle.com/ternaryrealm/lstm-time-series-explorations-with-keras

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = winm1.iloc[:,4].tail(1000)

plt.figure(figsize = (15, 5))
plt.plot(data, label = "Airline Passengers")

from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Activation, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle



data_raw = data.values.astype("float32")
data_raw = data_raw.reshape(-1, 1)
scaler = MinMaxScaler(feature_range = (0, 1))
dataset = scaler.fit_transform(data_raw)

TRAIN_SIZE = 0.80

train_size = int(len(dataset) * TRAIN_SIZE)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
print("Number of entries (training set, test set): " + str((len(train), len(test))))


def create_dataset(dataset, window_size = 1):
    data_X, data_Y = [], []
    for i in range(len(dataset) - window_size - 1):
        a = dataset[i:(i + window_size), 0]
        data_X.append(a)
        data_Y.append(dataset[i + window_size, 0])
    return(np.array(data_X), np.array(data_Y))

# Create test and training sets for one-step-ahead regression.
window_size = 2
train_X, train_Y = create_dataset(train, window_size)
test_X, test_Y = create_dataset(test, window_size)
print("Original training data shape:")
print(train_X.shape)

# Reshape the input data into appropriate form for Keras.
train_X = np.reshape(train_X, (train_X.shape[0], 1, train_X.shape[1]))
test_X = np.reshape(test_X, (test_X.shape[0], 1, test_X.shape[1]))
print("New training data shape:")
print(train_X.shape)

def fit_model(train_X, train_Y, window_size = 1):
    model = Sequential()

    model.add(LSTM(4,
                   input_shape = (1, window_size)))
    model.add(Dense(1))
    model.compile(loss = "mean_squared_error",
                  optimizer = "adam")
    model.fit(train_X,
              train_Y,
              epochs = 100,
              batch_size = 1,
              verbose = 2)

    return(model)

# Fit the first model.
model1 = fit_model(train_X, train_Y, window_size)

def predict_and_score(model, X, Y):
    # Make predictions on the original scale of the data.
    pred = scaler.inverse_transform(model.predict(X))
    # Prepare Y data to also be on the original scale for interpretability.
    orig_data = scaler.inverse_transform([Y])
    # Calculate RMSE.
    score = math.sqrt(mean_squared_error(orig_data[0], pred[:, 0]))
    return(score, pred)

rmse_train, train_predict = predict_and_score(model1, train_X, train_Y)
rmse_test, test_predict = predict_and_score(model1, test_X, test_Y)

print("Training data score: %.2f RMSE" % rmse_train)
print("Test data score: %.2f RMSE" % rmse_test)

# Start with training predictions.
train_predict_plot = np.empty_like(dataset)
train_predict_plot[:, :] = np.nan
train_predict_plot[window_size:len(train_predict) + window_size, :] = train_predict

# Add test predictions.
test_predict_plot = np.empty_like(dataset)
test_predict_plot[:, :] = np.nan
test_predict_plot[len(train_predict) + (window_size * 2) + 1:len(dataset) - 1, :] = test_predict


# Create the plot. RUN AT ONCE CTRL + RETURN
plt.figure(figsize = (15, 5))
plt.plot(scaler.inverse_transform(dataset), label = "True value")
plt.plot(train_predict_plot, label = "Training set prediction")
plt.plot(test_predict_plot, label = "Test set prediction")
plt.xlabel("Periodos")
plt.ylabel("$")
plt.title(os.path.basename(os.path.normpath(file)) + "Training data score: %.2f RMSE" % rmse_train + "  |   Test data score: %.2f RMSE" % rmse_test)
plt.legend()
plt.savefig(os.path.basename(os.path.normpath(file)) + '.png')
plt.show()
