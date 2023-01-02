import tensorflow.python.keras as k
k.backend.set_floatx("posit160")


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

df = pd.read_csv('housingdata.csv', header = None)

X = df.drop(13, axis = 1)
y = df[13]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

num_features = len(X_train[1,:])

model = k.models.Sequential()
model.add(k.layers.Dense(13, kernel_initializer = 'uniform', activation = 'tanh', input_dim = num_features)) 
model.add(k.layers.Dense(1, kernel_initializer = 'uniform'))
model.compile(optimizer = 'sgd', loss = 'mean_squared_error')
early_stopping_monitor = k.callbacks.EarlyStopping(monitor='val_loss', patience=30)
history_mse = model.fit(X_train, y_train, epochs = 100, callbacks = [early_stopping_monitor], verbose = 1, validation_split = 0.2) 

# EVALUATE MODEL IN THE TEST SET
score_mse_test = model.evaluate(X_test, y_test)
print('Test Score:', score_mse_test)

# EVALUATE MODEL IN THE TRAIN SET
score_mse_train = model.evaluate(X_train, y_train)
print('Train Score:', score_mse_train)