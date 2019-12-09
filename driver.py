import numpy as np
import pandas as pd

import train as trn
import model as mdl
import data_prep as dtp

from tensorflow.keras.utils import to_categorical

colnames = ['DATE', 'O', 'H', 'L', 'C', 'V1', 'V2']
win = pd.read_csv("data/winm1.csv", names=colnames, header=None)
wdo = pd.read_csv("data/wdom1.csv", names=colnames, header=None)

x = dtp.build_dataset(win, wdo, ROLL=100, thresh=10, debug=True, func='normalize')

y = np.array(x['target'])

del(x['target'])
x = np.array(x)

X_train = x
y_train = y

look_back = 100

x_train_reshaped, y_train_reshaped = dtp.timesteps(look_back, X_train, y_train)

y_train_one_hot = to_categorical(y_train_reshaped)

model = mdl.convo1D(look_back,
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
                    loss='categorical_crossentropy')

# model = mdl.lstm_model(1,
#                        look_back,
#                        activation1='relu',
#                        activation2='softmax',
#                        optimizer='adam',
#                        loss='categorical_crossentropy')


hist = trn.train_model(x_train_reshaped,
                       y_train_one_hot,
                       model,
                       name='convo',
                       tboard=True,
                       ckpt=False,
                       epochs=100,
                       batch=32,
                       )
