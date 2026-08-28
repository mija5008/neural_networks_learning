import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from keras.activations import relu, sigmoid
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
tf.__version__
data = np.random.randn(1000,150) # 1000 wierszy 150 kolumn
labels = np.random.randint(2,size = (1000, 1)) # tablica 1000x1 z losowymi liczbami z zakresu [0,2)
#czyli 0 lub 1  - etykiety klasyfikacji binarnej

# print(data.shape)
# print(labels.shape)
# print(data[:3])
# print(labels[:10])
# model = Sequential()
# model.add(Dense(units = 32, activation = 'relu', input_shape=(150,)))# Dense - warstwa gestopolaczona
# model.add(Dense(units = 1, activation = 'sigmoid'))
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
# model.fit(data, labels, epochs = 20 )
#######################            BATCH_SIZE
# model = Sequential()
# model.add(Dense(units = 32, activation = 'relu', input_shape=(150,)))# Dense - warstwa gestopolaczona
# model.add(Dense(units = 1, activation = 'sigmoid'))
# model.compile(optimizer='rmsprop',
#               loss='binary_crossentropy',
#               metrics=['accuracy'])
# model.fit(data, labels, epochs = 20, batch_size = 30 )
model = Sequential()
model.add(Dense(units = 32, activation = 'relu', input_shape=(150,)))# Dense - warstwa gestopolaczona
model.add(Dense(units = 1, activation = 'sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(data, labels, epochs = 20, batch_size = 30, validation_split = 0.2, verbose = 0 )

metrics = history.history
print(metrics.keys())
test_data = np.random.randn(5,150)
labels = np.random.randint(2,size = (5, 1)) # tablica 1000x1 z losowymi liczbami z zakresu [0,2)
probs = model.predict(test_data)
print(probs)
classes = (probs>0.5).astype(int)
print(classes)