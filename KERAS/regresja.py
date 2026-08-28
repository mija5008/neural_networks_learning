import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from keras.activations import relu, sigmoid
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
tf.__version__
data = np.random.random((1000,150)) # 1000 wierszy 150 kolumn
labels = 50*np.random.random(1000)
print(data[:3])
print(labels[:5])
model = Sequential()
model.add(Dense(units = 32, activation = 'relu', input_shape=(150,)))
model.add(Dense(units = 1,))
model.compile(optimizer = 'rmsprop',
              loss = 'mae',
              metrics = ['mse'])
model.fit(data, labels, epochs = 30, batch_size = 32, validation_split = 0.2)
test_data = np.random.random((1000,150))
probes = model.predict(test_data)
print(f'probes:{probes}')