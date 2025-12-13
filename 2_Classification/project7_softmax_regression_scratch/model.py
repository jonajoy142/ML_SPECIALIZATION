import numpy as np

def softmax(z):
    exp = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp / np.sum(exp, axis=1, keepdims=True)

class SoftmaxRegression:
    def __init__(self, lr=0.01, iterations=1000):
        self.lr = lr
        self.iterations = iterations
        self.W = None
        self.b = None

    def fit(self, X, y):
        m, n = X.shape
        k = len(np.unique(y))  # number of classes

        # One-hot encoding
        Y = np.zeros((m, k))
        Y[np.arange(m), y] = 1

        # Initialize parameters
        self.W = np.zeros((n, k))
        self.b = np.zeros((1, k))

        for _ in range(self.iterations):
            logits = np.dot(X, self.W) + self.b
            y_pred = softmax(logits)

            # Gradients
            dW = (1/m) * np.dot(X.T, (y_pred - Y))
            db = (1/m) * np.sum((y_pred - Y), axis=0, keepdims=True)

            # Update
            self.W -= self.lr * dW
            self.b -= self.lr * db

    def predict(self, X):
        logits = np.dot(X, self.W) + self.b
        y_pred = softmax(logits)
        return np.argmax(y_pred, axis=1)
