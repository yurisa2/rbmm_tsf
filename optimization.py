import numpy as np
import pandas as pd

import train as trn
import model as mdl
import data_prep as dtp

import os

import json
import requests

import itertools

from tensorflow.keras.utils import to_categorical

colnames = ['DATE', 'O', 'H', 'L', 'C', 'V1', 'V2']
win = pd.read_csv("data/winm1.csv", names=colnames, header=None, encoding='utf-16')
wdo = pd.read_csv("data/wdom1.csv", names=colnames, header=None, encoding='utf-16')

if os.path.exists('optimizations/params.csv') is False:

    opt_list = ['rmsprop', 'adam', 'nadam', 'Adadelta']
    act1_list = ['relu', 'elu', 'tanh']
    loss_list = ['binary_crossentropy', 'categorical_crossentropy']
    filters_list = [4, 8, 16, 32]
    kernel_list = [2, 3, 4, 5, 7]
    pool_list = [2, 3, 5]
    ndense_list = [4, 8, 16, 32, 64, 128, 256]
    roll_list = [20, 50, 100, 200]
    lb_list = [20, 50, 100, 200, 500]

    comb = list(itertools.product(opt_list,
                                  act1_list,
                                  loss_list,
                                  filters_list,
                                  kernel_list,
                                  pool_list,
                                  ndense_list,
                                  roll_list,
                                  lb_list
                                  )
                )

    param = pd.DataFrame(comb, columns=[
                                        'opt_list',
                                        'act1_list',
                                        'loss_list',
                                        'filters_list',
                                        'kernel_list',
                                        'pool_list',
                                        'ndense_list',
                                        'roll_list',
                                        'lb_list'
                                        ])
    print('Writing params.csv')
    param.to_csv('optimizations/params.csv')
else:
    param = pd.read_csv('optimizations/params.csv')

if os.path.exists('optimizations/optim_results.csv') is True:
    res_df = pd.read_csv('optimizations/optim_results.csv')
else:
    res_df = pd.DataFrame()

#################################
# Where the optimization STARTS #
#################################

hist = []
for index, row in param.sample(n=len(param)).iterrows():
    print('#############################################')
    if 'param_index' in res_df.columns:
        if index in res_df['param_index'].values:
            print('Passing by: ', index)
            continue

    params_online = pd.read_csv('http://ti.sa2.com.br/rbmm/optimizations/optim_results.csv')
    if index in params_online['param_index'].values:
        print('Passing by (online): ', index)
        continue


    print('Optimizing index: ', index)
    print('#############################################')

    x = dtp.build_dataset(win,
                          wdo,
                          ROLL=row["roll_list"],
                          thresh=10,
                          debug=False,
                          func='normalize')

    y = np.array(x['target'])

    del(x['target'])
    x = np.array(x)

    X_train = x
    y_train = y

    look_back = row['lb_list']

    x_train_reshaped, y_train_reshaped = dtp.timesteps(look_back,
                                                       X_train,
                                                       y_train)
    y_train_one_hot = to_categorical(y_train_reshaped)

    model = mdl.convo1D(look_back,
                        filters1=row["filters_list"],
                        filters2=row["filters_list"]*2,
                        kernel1=row["kernel_list"],
                        kernel2=row["kernel_list"],
                        pool_size1=row["pool_list"],
                        pool_size2=row["pool_list"],
                        ndense1=row["ndense_list"],
                        activation1=row["act1_list"],
                        activation2='softmax',
                        optimizer=row['opt_list'],
                        loss=row['loss_list']
                        )

    hist.append(trn.train_model(x_train_reshaped,
                                y_train_one_hot,
                                model,
                                name='convo',
                                tboard=False,
                                ckpt=False,
                                epochs=1000,
                                batch=100,
                                estop=True,
                                estop_patience=20,
                                )
                )

    hist_df = pd.DataFrame()
    for keys in hist[-1].history.keys():
        hist_df[keys] = (hist[-1].history[keys])
        pass
    hist_df['param_index'] = index

    headers = {'Content-Type': 'application/json',
               'Accept': 'application/json'}
    response = requests.post('http://ti.sa2.com.br/rbmm/update_csv.php',
                             data=json.dumps(hist_df.values.tolist()),
                             headers=headers
                             )

    res_df = res_df.append(hist_df)
    res_df.to_csv("optimizations/optim_results.csv", index=False)
