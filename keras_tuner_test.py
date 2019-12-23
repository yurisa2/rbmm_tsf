
import numpy as np
import pandas as pd

import os

import json
import requests

import itertools

from tensorflow.keras.utils import to_categorical

import numpy as np
import pandas as pd
from datetime import datetime
import os

from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Embedding, LSTM, GlobalMaxPooling1D, SpatialDropout1D
from tensorflow.keras.layers import MaxPooling1D, Conv1D, Flatten
from tensorflow.keras.callbacks import *
from tensorflow.keras.callbacks import ModelCheckpoint
from kerastuner.tuners import RandomSearch

def standardize(x):
    return ((x - np.mean(x)) / np.std(x))

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


def hours_standardize(date_in, func='standard'):
    hours = date_in.astype('datetime64[ns]').apply(lambda x: x.hour)
    if func == 'standard':
        hours = standardize(hours)
    else:
        hours = ((hours - np.min(hours)) / (np.max(hours) - np.min(hours)))

    return hours


def days_standardize(date_in, func='standard'):
    dayofweek = date_in.astype('datetime64[ns]').apply(lambda x: x.dayofweek)
    if func == 'standard':
        dayofweek = standardize(dayofweek)
    else:
        dayofweek = ((dayofweek - np.min(dayofweek)) / (np.max(dayofweek) - np.min(dayofweek)))

    return dayofweek


def result_functions(price_in, thresh_in):
    price_in = price_in.shift(-1)
    def result_func(last_2, thresh):
        raw = last_2[-1] - last_2[0]
        if raw > thresh:
        # if raw < (-1 * thresh):
            return 1
        # elif raw < (-1 * thresh):
        #     return 1
        else:
            return 0
    res = price_in.rolling(2).apply(result_func, args=(thresh_in,))
    return res


def build_dataset(dataset1, dataset2, ROLL, thresh, debug=False, func='standard', last_n=500):
    dataset1 = dataset1.iloc[-last_n:]
    dataset2 = dataset2.iloc[-last_n:]


    merge = pd.merge(dataset1, dataset2, on='DATE')

    definitive = pd.DataFrame(standardize_col_roll(merge['C_x'], ROLL, func))
    definitive['c2'] = standardize_col_roll(merge['C_y'], ROLL, func)
    definitive['v11'] = standardize_col_roll(merge['V1_x'], ROLL, func)
    definitive['v12'] = standardize_col_roll(merge['V2_x'], ROLL, func)
    definitive['v21'] = standardize_col_roll(merge['V1_y'], ROLL, func)
    definitive['v22'] = standardize_col_roll(merge['V2_y'], ROLL, func)
    definitive['hours'] = hours_standardize(merge['DATE'], func)
    definitive['dow'] = days_standardize(merge['DATE'], func)
    definitive['target'] = pd.Categorical(result_functions(merge['C_x'], thresh))
    definitive = definitive.dropna()

    if debug is True:
        merge.to_csv("debug-raw-merge.csv")
        definitive.to_csv("debug-raw-definitive.csv")
        pass
    return definitive


def timesteps(look_back, X_train, y_train):
    nb_samples = X_train.shape[0] - look_back
    x_train_reshaped = np.zeros((nb_samples, look_back, 8))
    y_train_reshaped = np.zeros((nb_samples))
    for i in range(nb_samples):
        y_position = i + look_back
        x_train_reshaped[i] = X_train[i:y_position]
        y_train_reshaped[i] = y_train[y_position]
    return x_train_reshaped, y_train_reshaped



def lstm_model(units,
               look_back,
               activation1='relu',
               activation2='softmax',
               optimizer='adam',
               loss='categorical_crossentropy'
               ):

    model = Sequential()

    model.add(LSTM(units, activation=activation1, input_shape=np.int32((look_back, 8))))
    model.add(Dense(2, activation=activation2))
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    model.summary()

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
    # with strategy.scope():
    model = Sequential()  # Create the architecture
    model.add(Conv1D(filters1,
                    (kernel1),
                    activation='relu',
                    input_shape=(look_back, 8)))
    model.add(MaxPooling1D(pool_size=(pool_size1)))
    model.add(Conv1D(filters2, (kernel2), activation='relu'))
    model.add(MaxPooling1D(pool_size=(pool_size2)))
    model.add(Flatten())
    model.add(Dense(np.int32(ndense1), activation=activation1))
    model.add(Dense(2, activation=activation2))
    model.compile(loss=loss,
                  optimizer=optimizer,
                  metrics=['accuracy','categorical_accuracy'])
    model.summary()
    return model


def train_model(x,
                y,
                model,
                name="noname",
                tboard=False,
                ckpt=False,
                epochs=10,
                batch=3,
                val_split=0.3,
                estop=False,
                estop_patience=10,
                estop_min_delta=0.0001,
                estop_monitor='val_acc',
                steps=46,
                cweights=0

                ):

    from datetime import datetime
    import os
    from tensorflow.keras.callbacks import ModelCheckpoint
    from tensorflow.keras.callbacks import TensorBoard
    from tensorflow.keras.callbacks import EarlyStopping

    callbacks = []


    if estop is True:
        earlystop_callback = EarlyStopping(
          monitor=estop_monitor, min_delta=estop_min_delta,
          patience=estop_patience)
        callbacks.append(earlystop_callback)
        pass


    if tboard is True:
        logdir = 'logs/' + datetime.now().strftime("%Y%m%d-%H%M%S-") + name
        if not os.path.exists('logs'):
            os.makedirs('logs')

        os.mkdir(logdir)
        logdir = os.path.join(logdir)

        tensorboard_callback = TensorBoard(log_dir=logdir,
                                           histogram_freq=1,
                                          #  profile_batch=100000000
                                           )

        callbacks.append(tensorboard_callback)
        pass

    if ckpt is True:

        if not os.path.exists('checkpoints'):
            os.makedirs('checkpoints')

        ckpf_dir = os.path.join('checkpoints',
                            datetime.now().strftime("%Y%m%d-%H%M%S-") + name,
                            )
        os.makedirs(ckpf_dir)

        ckpf = os.path.join('checkpoints',
                            datetime.now().strftime("%Y%m%d-%H%M%S-") + name,
                            name + '.hdf5')

        checkpointer = ModelCheckpoint(filepath=ckpf,
                                       verbose=1,
                                       save_best_only=True)

        callbacks.append(checkpointer)
        pass

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=val_split)

    X_train = X_train.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_train = y_train.astype(np.float32)
    y_test = y_test.astype(np.float32)

    hist = model.fit(X_train,
                     y_train,
                     batch_size=batch,
                     epochs=epochs,
                    #  validation_split=val_split,
                     validation_data=(X_test, y_test),
                     callbacks=callbacks,
                    #  steps_per_epoch=steps
                     class_weight=cweights
                     )
    return hist


colnames = ['DATE', 'O', 'H', 'L', 'C', 'V1', 'V2']
win = pd.read_csv("https://github.com/yurisa2/rbmm_tsf/raw/master/data/winm1.csv", names=colnames, header=None, encoding='utf-16')
wdo = pd.read_csv("https://github.com/yurisa2/rbmm_tsf/raw/master/data/wdom1.csv", names=colnames, header=None, encoding='utf-16')

def full_datasets(dataset1,
                  dataset2,
                  ROLL,
                  thresh,
                  debug,
                  func,
                  look_back):

    x = build_dataset(dataset1,
                        dataset2,
                        ROLL=ROLL,
                        thresh=thresh,
                        debug=debug,
                        func=func)


    y = np.array(x['target'])

    del(x['target'])
    x = np.array(x)

    return x, y

"""Categorical Shit"""

def categorize(look_back, x, y):
    x_train_reshaped, y_train_reshaped = timesteps(look_back, x, y)
    y_train_one_hot = to_categorical(y_train_reshaped)

    neg, pos = np.bincount(pd.Categorical(y))
    total = neg + pos
    print('Examples:\n    Total: {}\n    Positive: {} ({:.2f}% of total)\n'.format(
      total, pos, 100 * pos / total))

    # Scaling by total/2 helps keep the loss to a similar magnitude.
    # The sum of the weights of all examples stays the same.
    weight_for_0 = (1 / neg)*(total)/2.0
    weight_for_1 = (1 / pos)*(total)/2.0

    class_weight = {0: weight_for_0, 1: weight_for_1}

    print('Weight for class 0: {:.2f}'.format(weight_for_0))
    print('Weight for class 1: {:.2f}'.format(weight_for_1))

    return class_weight

"""# Prepara E Treina Modelo"""

def build_model(hp):

    model =  lstm_model(hp.Int('units',
                                        min_value=4,
                                        max_value=512,
                                        step=32),
                      look_back,
                      activation1='relu',
                      activation2='softmax',
                      optimizer='adam',
                      loss='categorical_crossentropy'
                      )
    return model
