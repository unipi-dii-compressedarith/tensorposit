import tensorflow.python.keras as k
k.backend.set_floatx("posit160")

import pandas as pd 
import numpy as np 
from sklearn.model_selection import train_test_split
from sklearn import neural_network
from sklearn import  metrics
import matplotlib.pyplot as plt


train = pd.read_csv("train-mnist.csv")

Y = train['label'][:2000] # use more number of rows for more training 
X = train.drop(['label'], axis = 1)[:2000] # use more number of rows for more training 
x_train, x_val, y_train, y_val = train_test_split(X.values, Y.values, test_size=0.10, random_state=42)

# network parameters 
batch_size = 128
num_classes = 10
epochs = 5 # Further Fine Tuning can be done

# input image dimensions
img_rows, img_cols = 28, 28

# preprocess the train data 
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_train = x_train.astype('float32')
x_train /= 255

# preprocess the validation data
x_val = x_val.reshape(x_val.shape[0], img_rows, img_cols, 1)
x_val = x_val.astype('float32')
x_val /= 255

input_shape = (img_rows, img_cols, 1)

# convert the target variable 
y_train = k.utils.np_utils.to_categorical(y_train, num_classes)
y_val = k.utils.np_utils.to_categorical(y_val, num_classes)


model = k.models.Sequential()
# model.add(k.layers.Conv2D(16, kernel_size=(3, 3), activation='tanh', input_shape=input_shape))
model.add(k.layers.Flatten(input_shape=input_shape))
model.add(k.layers.Dense(128, activation='tanh'))
# model.add(k.layers.BatchNormalization())
# model.add(k.layers.Dropout(0.05))
model.add(k.layers.Dense(num_classes, activation='softmax'))

model.compile(loss=k.losses.categorical_crossentropy,  optimizer='adam', metrics=['accuracy'])

earlystop_callback = k.callbacks.EarlyStopping(monitor='val_accuracy',patience=25)

history = model.fit(x_train, y_train, batch_size=batch_size, epochs=50, verbose=1, validation_data=(x_val, y_val), callbacks=earlystop_callback)


def plot(history):
  acc = history.history['accuracy']
  loss = history.history['loss']
  epochs = range(len(acc))
  plt.figure()
  plt.plot(epochs, acc, 'bo', label='Training acc')
  plt.title('Training accuracy')
  plt.legend()
  #plt.savefig('ac.png')
  plt.figure()
  plt.plot(epochs, loss, 'bo', label='Training loss')
  plt.title('Training loss')
  plt.legend()
  #plt.savefig('loss.png')
  plt.show()

