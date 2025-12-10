import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))


class LogisticRegressionScratch:
    def __init__(self, lr=0.01, iterations=1000):
        self.lr = lr
        self.iterations = iterations

    def fit(self, X, y):
        m, n = X.shape
        self.theta = np.zeros(n)
        self.bias = 0

        for _ in range(self.iterations):
            linear = np.dot(X, self.theta) + self.bias
            y_pred = sigmoid(linear)

            d_theta = (1/m) * np.dot(X.T, (y_pred - y))
            d_bias = (1/m) * np.sum(y_pred - y)

            self.theta -= self.lr * d_theta
            self.bias -= self.lr * d_bias

    def predict_proba(self, X):
        return sigmoid(np.dot(X, self.theta) + self.bias)

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)
