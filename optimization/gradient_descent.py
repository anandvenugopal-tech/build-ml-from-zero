"""
Gradient Descent from Scratch

This module implements the gradient descent algorithm to minimize Mean Squared Error (MSE)
loss for a simple linear regression model.

"""

#import numpy library
import numpy as np

#define a class gradient descent.
class GradientDescent:
    def __init__(self, learning_rate = 0.01, n_iter = 1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
        self.loss_history = []

    #define loss function
    def _mse_loss(self, y, y_pred):
        return np.mean((y_pred - y) ** 2)
    
    def fit(self, x, y): 
        n_samples, n_features = x.shape
        self.weights = np.zeros(n_samples)


