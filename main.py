import tensorflow as tf
import numpy as np
import pandas as pd
import plotly.express as px
tf.__version__
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.layers import Activation
from tensorflow.keras.activations import linear, sigmoid, relu, softmax
# model = Sequential()
# #print(model)
 # warstwa wejsciowa 4 neurony(units) 10 wejsc
#
# random_data  =sorted(np.random.randn(400))
# data = pd.DataFrame({'data': random_data,'linear':linear(random_data)})
# dataS = pd.DataFrame({'data': random_data,'sigmoid':sigmoid(random_data)})
# dataR = pd.DataFrame({'data': random_data,'relu':relu(random_data)})
# print(data.head())
# print(dataS.head())
# print(dataR.head())
#
# fig = px.line(dataS,x='data',y='sigmoid', width = 800, range_y=[-0.5, 2])
# figR = px.line(dataR,x='data',y='relu', width = 800, range_y=[-0.5, 2])
# #fig.show()
# #figR.show()
# #model.summary() # podsumowanie , param: d = (10+1)*4, d_1 = (4+1)*2, total = d +d_1 = 54
model1 = Sequential()
model1.add(Dense(units=8,activation= 'relu', input_shape=(10,)))
model1.add(Dense(units=1,activation= 'sigmoid'))
model1.summary()
model = Sequential()
model.add(Dense(units=8, activation='relu', input_shape=(10,)))
model.add(Dense(units=1, activation='sigmoid'))
model.summary()
#klasyfikacja binarna
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# klasyfikacja wieloklasowa
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# regresja
model.compile(optimizer='rmsprop',
              loss='mse')