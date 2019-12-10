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

hist = []

opt_list = ['rmsprop','adam','nadam','Adadelta']
act1_list = ['relu','elu']
loss_list = ['binary_crossentropy','categorical_crossentropy']
filters_list = [4,8,16,32]
kernel_list = [2,3,4,5,7]
pool_list = [2,3,5]
ndense_list = [16,32,64,128,256]


for optimizer_for in opt_list:
    for act1_for in act1_list:
        for loss_for in loss_list:
            for filters_for in filters_list:
                for kernel_for in kernel_list:
                    for pool_for in pool_list:
                        for ndense_for in ndense_list:
                            
                    
                            model = mdl.convo1D(look_back,
                                                filters1=filters_for,
                                                filters2=filters_for*2,
                                                kernel1=kernel_for,
                                                kernel2=kernel_for,
                                                pool_size1=pool_for,
                                                pool_size2=pool_for,
                                                ndense1=ndense_for,
                                                activation1=act1_for,
                                                activation2='softmax',
                                                optimizer=optimizer_for,
                                                loss=loss_for)
                            
                            # model = mdl.lstm_model(1,
                            #                        look_back,
                            #                        activation1='relu',
                            #                        activation2='softmax',
                            #                        optimizer='adam',
                            #                        loss='categorical_crossentropy')
                            
                             
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
