# from datetime import datetime
# import os
import pandas as pd
import numpy as np


# Merge with other symbols,
# Result Function


def standardize(x):
    return ((x - np.mean(x)) / np.std(x))


def standardize_col_roll(col, window=100):

    def standardize_rolling(x):
        return ((x - np.mean(x)) / np.std(x))[-1]

    col_standardized = col.rolling(window).apply(standardize_rolling)
    return col_standardized


def hours_days_standardize(date_in):
    hours = date_in.astype('datetime64[ns]').apply(lambda x: x.hour)
    hours = standardize(hours)

    dayofweek = date_in.astype('datetime64[ns]').apply(lambda x: x.dayofweek)
    dayofweek = standardize(dayofweek)
    return hours, dayofweek


def result_functions(price_in, thresh_in):
    price_in = price_in.shift(-1)
    def result_func(last_2, thresh=20):
        raw = last_2[-1] - last_2[0]
        if raw > thresh:
            return 2
        elif raw < (-1 * thresh):
            return 1
        else:
            return 0
    res = price_in.rolling(2).apply(result_func, args=(thresh_in,))
    return res


def build_dataset(dataset1, dataset2, ROLL=100, thresh=100):

    merge = pd.merge(dataset1, dataset2, on='DATE')

    definitive = pd.DataFrame(standardize_col_roll(merge['C_x'], ROLL))
    definitive['c2'] = standardize_col_roll(merge['C_y'], ROLL)
    definitive['v11'] = standardize_col_roll(merge['V1_x'], ROLL)
    definitive['v12'] = standardize_col_roll(merge['V2_x'], ROLL)
    definitive['v21'] = standardize_col_roll(merge['V1_y'], ROLL)
    definitive['v22'] = standardize_col_roll(merge['V2_y'], ROLL)
    definitive['hours'], merge['dow'] = hours_days_standardize(merge['DATE'])
    definitive['target'] = pd.Categorical(result_functions(merge['C_x'], thresh))
    definitive = definitive.dropna()
    return definitive


def timesteps(look_back, X_train, y_train):
    nb_samples = X_train.shape[0] - look_back
    x_train_reshaped = np.zeros((nb_samples, look_back, 7))
    y_train_reshaped = np.zeros((nb_samples))
    for i in range(nb_samples):
        y_position = i + look_back
        x_train_reshaped[i] = X_train[i:y_position]
        y_train_reshaped[i] = y_train[y_position]
    return x_train_reshaped, y_train_reshaped
