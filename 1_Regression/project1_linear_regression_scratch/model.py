# model.py
import numpy as np

class LinearRegressionScratch:
    def __init__(self, learning_rate=0.001, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.theta = None
        self.bias = None
        self.cost_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # initialize parameters
        self.theta = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.iterations):
            y_pred = np.dot(X, self.theta) + self.bias

            # gradients
            d_theta = (1/n_samples) * np.dot(X.T, (y_pred - y))
            d_bias  = (1/n_samples) * np.sum(y_pred - y)

            # update
            self.theta -= self.learning_rate * d_theta
            self.bias  -= self.learning_rate * d_bias

            # record cost
            cost = np.mean((y_pred - y) ** 2)
            self.cost_history.append(cost)

    def predict(self, X):
        return np.dot(X, self.theta) + self.bias
