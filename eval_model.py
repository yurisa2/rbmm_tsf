#
# hist.history.get()
#
# import matplotlib.pyplot as plt
# # Plot training & validation accuracy values
# plt.plot(hist.history['acc'])
# plt.plot(hist.history['val_acc'])
# plt.plot(hist.history['val_categorical_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
#
# # Plot training & validation loss values
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()

import tensorflow.keras
from keras.models import load_model
model = load_model('model.h5')


import tensorflow as tf
classifierLoad = tf.keras.models.load_model('model.h5')



from sklearn.model_selection import train_test_split
xTrain, xTest, yTrain, yTest = train_test_split(x_train_reshaped, y_train_reshaped, test_size = 0.3, random_state = 0)

y_pred = model.predict_classes(xTest)

import tensorflow as tf
con_mat = tf.math.confusion_matrix(labels=yTest, predictions=y_pred) # .numpy()

print(con_mat)


with tf.Session():
   print('Confusion Matrix: \n\n', tf.Tensor.eval(con_mat,feed_dict=None, session=None))
