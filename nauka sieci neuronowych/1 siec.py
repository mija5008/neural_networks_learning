import numpy as np
import pandas as pd

np.random.seed(42)

X = np.array([0.7, 1.4])
y_true = np.array([1.8])

def initialize_parameters (n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x)
    W2 = np.random.randn(n_h, n_y)
    return W1, W2

def forward_propagation(X, W1, W2):
    H1 = np.dot(X, W1)
    y_pred = np.dot(H1, W2)
    return H1, y_pred

def calculate_error(y_pred, y_true):
    return y_pred - y_true

def predict(X, W1, W2):
    _, y_pred = forward_propagation(X, W1, W2)
    return y_pred[0]

def backpropagation(X, W1, W2, learning_rate, iters = 1000, precision =0.0000001 ):
    H1, y_pred = forward_propagation(X, W1, W2)
    train_loss = []
    for i in range(iters):
        error = calculate_error(y_pred, y_true)
        W2 = W2 - learning_rate * error* H1.T
        W1 = W1 - learning_rate * error* np.dot(X.T, W2.T)

        y_pred = predict(X, W1, W2)
        print(f'Iter #{i}: y_pred {y_pred}: loss: {abs(calculate_error(y_pred, y_true[0]))}')
        train_loss.append(abs(calculate_error(y_pred, y_true[0])))
        if abs(error) < precision:
                break
    return W1, W1, train_loss

def build_model():
    W1, W2 = initialize_parameters(2,2,1)
    W1,W2, train_loss = backpropagation(X, W1, W2, 0.01)
    model = {'W1': W2, 'W2': W1, 'train_loss': train_loss}
    return model
model = build_model()

