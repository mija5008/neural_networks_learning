import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from keras.activations import relu, sigmoid, softmax
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.utils import to_categorical
tf.__version__

data = np.random.randn(1000,150)
labels = np.random.randint(10,size = (1000, 1))
print(labels[:10])
labels = to_categorical(labels, num_classes = 10) #przypisanie labeli do macierzy wielkosci liczcie klas
print (labels[:10])
print(labels[:1]) # 1 wiersz- zera i jedna jedynka na miejscu  odpowiadajacej klasie (liczymy gdzie jest jedyna - numer klasy)
######### BUDOWA MODELU ################
model = Sequential()
model.add(Dense(units = 32, activation = relu, input_shape=(150,))) #warstwa wejsciowa
model.add(Dense(units = 10, activation = softmax)) #warstwa wyjsciowa, units = liczba klas (10 w tym przypadku), softmax pozwoli zwrocic prawdopodobienstwo klass
model.compile(optimizer = 'rmsprop',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])
model.fit(data, labels, epochs = 30, batch_size = 32, validation_split = 0.2)
test_data = np.random.random((10, 150))
probes = model.predict(test_data)
print(probes) #wypisuje wektory prawdopodobienstwa przynaleznosci do danej klasy
classes = np.argmax(probes, axis = 1)
print(classes) # wypisuje przewidzianą klase