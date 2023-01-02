import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import tensorflow.python.keras as k
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

k.backend.set_floatx("posit160")


data = pd.read_csv("iap.csv", usecols = [1], engine = "python", skipfooter = 3)

data_raw = data.values.astype("float32")
scaler = MinMaxScaler(feature_range = (0, 1))
dataset = scaler.fit_transform(data_raw)

def get_data(data, look_back):
    dataX, dataY = [], []
    for i in range(len(data)-look_back-1):
        a = data[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(data[i+look_back, 0])
    return np.array(dataX), np.array(dataY)

look_back = 1
X_train, y_train = get_data(dataset, look_back)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

classifier = k.models.Sequential()
classifier.add(k.layers.LSTM(5, input_shape = (1, look_back)))
classifier.add(k.layers.Dense(1))
classifier.compile(loss = "mean_squared_error", optimizer = "adam")
history = classifier.fit(X_train, y_train, epochs = 100, batch_size = 1, verbose = 2)
