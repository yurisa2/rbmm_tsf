# from datetime import datetime
# import os
import pandas as pd
import numpy as np

colnames = ['DATE', 'O', 'H', 'L', 'C', 'V1', 'V2']
winm1 = pd.read_csv("data/winm1.csv", names=colnames, header=None)

# Merge with other symbols,
# Result Function


def standardize(x):
    return ((x - np.mean(x)) / np.std(x))


def standardize_col_roll(col, window=100):

    def standardize_rolling(x):
        return ((x - np.mean(x)) / np.std(x))[-1]

    col_standardized = col.rolling(window).apply(standardize_rolling)
    return col_standardized


hours = winm1['DATE'].astype('datetime64[ns]').apply(lambda x: x.hour)
hours = standardize(hours)

dayofweek = winm1['DATE'].astype('datetime64[ns]').apply(lambda x: x.dayofweek)
dayofweek = standardize(dayofweek)

data = [hours, dayofweek,
        standardize_col_roll(winm1['C'], 5),
        standardize_col_roll(winm1['V1'], 5),
        standardize_col_roll(winm1['V2'], 5)
        ]

df = pd.DataFrame(data)
df = df.transpose()
df.columns = ["Team", "Player", "Salary", "Role"]
