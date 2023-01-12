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

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)

X, y = split_sequence(dataset, 6)
X = X.reshape((-1, 6, 1))

classifier = k.models.Sequential()
classifier.add(k.layers.Conv1D(filters=128, kernel_size=2, activation='tanh', input_shape=X.shape[1:]))
classifier.add(k.layers.MaxPooling1D(pool_size=2))
classifier.add(k.layers.Flatten())
classifier.add(k.layers.Dense(1))
classifier.compile(loss = "mean_squared_error", optimizer = "adam")
history = classifier.fit(X_train, y_train, epochs = 100, batch_size = 1, verbose = 2)



