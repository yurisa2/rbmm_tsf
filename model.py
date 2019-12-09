import numpy as np
import pandas as pd
from datetime import datetime
import os

import train as trn

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, GlobalMaxPooling1D, SpatialDropout1D
from tensorflow.keras.layers import MaxPooling1D, Conv1D, Flatten
from tensorflow.keras.callbacks import *
from tensorflow.keras.callbacks import ModelCheckpoint


def lstm_model(units,
               look_back,
               activation1='relu',
               activation2='softmax',
               optimizer='adam',
               loss='categorical_crossentropy'
               ):
    model = Sequential()

    model.add(LSTM(units, activation=activation1, input_shape=(look_back, 7)))
    model.add(Dense(3, activation=activation2))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    # model.summary()

    return model


def convo1D(look_back,
            filters1=8,
            filters2=16,
            kernel1=5,
            kernel2=5,
            pool_size1=2,
            pool_size2=2,
            ndense1=16,
            activation1='relu',
            activation2='softmax',
            optimizer='adam',
            loss='categorical_crossentropy'):

    model = Sequential()  # Create the architecture
    model.add(Conv1D(filters1,
                     (kernel1),
                     activation='relu',
                     input_shape=(look_back, 7)))
    model.add(MaxPooling1D(pool_size=(pool_size1)))
    model.add(Conv1D(filters2, (kernel2), activation='relu'))
    model.add(MaxPooling1D(pool_size=(pool_size2)))
    model.add(Flatten())
    model.add(Dense(ndense1, activation=activation1))
    model.add(Dense(3, activation=activation2))
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy','categorical_accuracy'])
    return model
