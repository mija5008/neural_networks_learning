import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow import keras
from keras.activations import relu, sigmoid, softmax
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets.mnist import load_data
from tensorflow.keras.utils import to_categorical
import json
from tensorflow.keras.models import  model_from_json


tf.__version__

(X_train, y_train), (X_test, y_test) = load_data() # ładujemy danes

print(f'X_train shape: {X_train.shape}')
print(f'y_train shape: {y_train.shape}')
print(f'X_test shape: {X_test.shape}')
print(f'y_test shape: {y_test.shape}')

X_train = X_train/255.0
X_test = X_test/255.0

model = Sequential()
model.add(Flatten(input_shape=(28,28))) #warstwa flatten wyplaszcza dane
# zdjecie 28x28 przedsrawia jako wektor o dlugosci 784
model.add(Dense(units = 128, activation = 'relu') )
model.add(Dropout(0.2))#określa % neuronow w warstwie ktore chcielibysmy pominać (aby uniknąć przeuczenia)
model.add(Dense(units = 10, activation = 'softmax') ) # 10 wyjsc = 10 neuronów = 10 klas

model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
y_train = to_categorical(y_train, num_classes = 10)
model.fit(X_train, y_train, epochs = 10, batch_size = 32, validation_split = 0.2 )
print( f'lista warstw:{model.layers}') # lista warstw
print(f' lista wejsc {model.inputs}') # lista wejsc (tensorów)
print(f' lista wyjsc {model.outputs}') # lista wyjsc (tensorów)
print(f' slownik z configiem{model.get_config()}') # słownik z konfiguracją modelu
print(f'lista wag {model.get_weights()}') # słownik z listą tensorów wszystkich wag modelu
print(f'rozmiar (macierz) wag pierwszego elementu {model.get_weights()[0].shape}') # i tak mozna z [1],[2] itd

#model do jsona
model_json = model.to_json()
parsed = json.loads(model_json)
print(json.dumps(parsed, indent=4))
# model z jsona
model2 = model_from_json(model_json)
model2.summary