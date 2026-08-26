import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
import math
#--------------------------------RELU-------------------------
# np.set_printoptions(precision=6)
# def relu_max(x):
#     return max(0.0, x)
#
# data = np.random.randn(50)
# data = sorted(data)
# print(data)
#
# max_relu_data = np.array([relu_max(x) for x in data])
# print(max_relu_data)
#
# df = pd.DataFrame( {'data': data,'max_relu_data': max_relu_data})
# fig = px.line(df, x='data', y='max_relu_data',width=600, height=400, title='Max Relu Data')
# fig.show()
# #-      --------------SIGMOID--------------------------#
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
#
# data= 3*np.random.randn(50)
# data = sorted(data)
#
#
# sigmoid_data = np.array([sigmoid(x) for x in data])
# print(sigmoid_data)
#
# df = pd.DataFrame({'data': data,"sigmoid_data": sigmoid_data})
# print(df)
# fig = px.line(df, x="data", y="sigmoid_data", width=800, height=400, title= 'sigmoid data')
# fig.show()
# def tanh(x):
#     return np.tanh(x)
# data = np.random.randn(50)
# data = sorted(data)
# print(data)
# data_tanh = np.array([tanh(x) for x in data ])
# print(data_tanh)
# df = pd.DataFrame({'data': data, "tanh_data": data_tanh})
# print(df)
# fig = px.line(df, x="data", y="tanh_data", width=800, height=400, title= 'tanh data')
# fig.show()
# def softmax(x):
#     return np.exp(x) / np.sum(np.exp(x))
# data = np.random.randn(4,5)
# print(softmax(data))
# result = softmax(data)
# print(result.sum(axis=1))
#function = lambda w: 2**w - 4w
# max_iters = 1000
# iters = 0
# w_0 = -1
# learning_rate = 0.01
# previous_step_size = 1
# precision = 0.000001
# derivative = lambda w: 2 * w - 4
# points = []
# while previous_step_size > precision and iters < max_iters:
#     w_prev = w_0
#     w_0 = w_0 - learning_rate * derivative(w_prev)
#     previous_step_size = abs(w_0 - w_prev)
#     iters += 1
#     points.append(w_0)
#     print(f'iteracje : {iters}, obecny punkt: {w_0}')
# print(f'minimum lokalne w punkcie :{w_0}')