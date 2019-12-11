import numpy as np
import pandas as pd

import train as trn
import model as mdl
import data_prep as dtp

import itertools

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

hist = []

opt_list = ['rmsprop', 'adam', 'nadam', 'Adadelta']
act1_list = ['relu', 'elu']
loss_list = ['binary_crossentropy', 'categorical_crossentropy']
filters_list = [4, 8, 16, 32]
kernel_list = [2, 3, 4, 5, 7]
pool_list = [2, 3, 5]
ndense_list = [16, 32, 64, 128, 256]

test = list(itertools.product(opt_list,
                              act1_list,
                              loss_list,
                              filters_list,
                              kernel_list,
                              pool_list,
                              ndense_list))

param = pd.DataFrame(test, columns=[
                     'opt_list',
                     'act1_list',
                     'loss_list',
                     'filters_list',
                     'kernel_list',
                     'pool_list',
                     'ndense_list'])

for opt_param in param:
    model = mdl.convo1D(look_back,
                        filters1=opt_param["filters_list"],
                        filters2=opt_param["filters_list"]*2,
                        kernel1=opt_param["kernel_list"],
                        kernel2=opt_param["kernel_list"],
                        pool_size1=opt_param["pool_list"],
                        pool_size2=opt_param["pool_list"],
                        ndense1=opt_param["ndense_list"],
                        activation1=opt_param["act1_list"],
                        activation2='softmax',
                        optimizer='opt_list',
                        loss='loss_list')

    hist.append(trn.train_model(x_train_reshaped,
                                y_train_one_hot,
                                model,
                                name='convo',
                                tboard=False,
                                ckpt=False,
                                epochs=1000,
                                batch=100,
                                estop=True,
                                estop_patience=10,
                                )
                )

model.get_config()
