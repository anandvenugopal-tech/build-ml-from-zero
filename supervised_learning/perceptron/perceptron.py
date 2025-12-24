"""
Perceptron Algorithm (From Scratch)
----------------------------------

This module implements the classical Perceptron algorithm for
binary classification.

The Perceptron is the simplest neural network:
    - No hidden layers
    - Linear decision boundary
    - Uses step activation function

Training rule:
    w = w + learning_rate * (y - y_pred) * x
    b = b + learning_rate * (y - y_pred)
"""

import numpy as np


class Perceptron:

    def __init__(self, learning_rate=0.01, n_iter=1000):
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.weights = None
        self.bias = None

    def _activation(self, z):
        """
        Step activation function.

        If z >= 0 → 1  
        If z < 0  → 0
        """
        return np.where(z >= 0, 1, 0)

    def fit(self, X, y):

        n_samples, n_features = X.shape

        # Initialize weights and bias
        self.weights = np.zeros(n_features)
        self.bias = 0.0

        # Convert labels to {0, 1} if they are {-1, 1}
        y = np.where(y <= 0, 0, 1)

        for _ in range(self.n_iter):

            for idx in range(n_samples):
                # Linear model
                z = np.dot(X[idx], self.weights) + self.bias

                # Activation (binary prediction)
                y_pred = self._activation(z)

                # Update rule if prediction is wrong
                update = self.learning_rate * (y[idx] - y_pred)

                self.weights += update * X[idx]
                self.bias += update

    def predict(self, X):
        """
        Predict class labels (0 or 1).
        """
        z = np.dot(X, self.weights) + self.bias
        return self._activation(z)
        

if __name__ == "__main__":
    # Simple OR dataset
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])

    y = np.array([0, 1, 1, 1])  # OR logic

    model = Perceptron(learning_rate=0.1, n_iter=10)
    model.fit(X, y)

    print("Weights:", model.weights)
    print("Bias:", model.bias)
    print("Predictions:", model.predict(X))

