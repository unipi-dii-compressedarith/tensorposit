from numpy import loadtxt
import tensorflow.python.keras as k
k.backend.set_floatx("posit160")

# load the dataset
dataset = loadtxt('pima-indians-diabetes.data.csv', delimiter=',')
# split into input (X) and output (y) variables
X = dataset[:,0:8]
y = dataset[:,8]

# define the keras model
model = k.models.Sequential()
model.add(k.layers.Dense(12, input_shape=(8,), kernel_initializer=k.initializers.RandomNormal(stddev=0.01), activation='leaky_relu'))
model.add(k.layers.Dense(8, activation='leaky_relu'))
model.add(k.layers.Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X, y, epochs=100, batch_size=10)

#print_weights = k.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: print(model.layers[0].get_weights()))

#model.fit(X, y, epochs=50, batch_size=10, callbacks = [print_weights])