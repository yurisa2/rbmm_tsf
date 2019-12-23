import numpy as np
import pandas as pd

import train as trn
import model as mdl
import data_prep as dtp

from tensorflow.keras.utils import to_categorical

colnames = ['DATE', 'O', 'H', 'L', 'C', 'V1', 'V2']
win = pd.read_csv("data/winm1.csv", names=colnames, header=None, encoding='utf-16')
wdo = pd.read_csv("data/wdom1.csv", names=colnames, header=None, encoding='utf-16')


x = dtp.build_dataset(win, wdo, ROLL=500, thresh=10, debug=True, func='normalize')


y = np.array(x['target'])

del(x['target'])
x = np.array(x)

look_back = np.int64(50)

x_train_reshaped, y_train_reshaped = dtp.timesteps(look_back, x, y)
y_train_one_hot = to_categorical(y_train_reshaped)

del(x)
del(y)



from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
X_train, X_test, y_train, y_test = train_test_split(x_train_reshaped, y_train_reshaped, test_size=0.3)


from tensorflow.keras.models import load_model
model = load_model('4876.hdf5')


Y_pred = model.predict(X_test)
y_pred = np.argmax(Y_pred, axis=1)

y_test.shape
y_pred.shape

results = pd.DataFrame()

results["test"] = y_test
results["pred"] = y_pred

print('Confusion Matrix')
print(confusion_matrix(y_test, y_pred))
print('Classification Report')
target_names = ['0', '1', '2']
print(classification_report(y_test, y_pred, target_names=target_names))

rig = confusion_matrix(y_test, y_pred)[1][1] + confusion_matrix(y_test, y_pred)[2][2]
wro = confusion_matrix(y_test, y_pred)[1][2] + confusion_matrix(y_test, y_pred)[2][1]
