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
                            epochs=5,
                            batch=150,
                            estop=True,
                            estop_patience=10,
                            )
            )
count = count + 1









#
#
# model.get_config()
#
#
# con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
#
# con_mat_df = pd.DataFrame(con_mat_norm,
#                      index = classes,
#                      columns = classes)
#
#
#
# y_pred=model.predict_classes(test_images)
# con_mat = tf.math.confusion_matrix(labels=y_true, predictions=y_pred).numpy()
#
