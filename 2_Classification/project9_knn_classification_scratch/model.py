import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def _distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _predict_single(self, x):
        distances = []

        for i, x_train in enumerate(self.X_train):
            dist = self._distance(x, x_train)
            distances.append((dist, self.y_train[i]))

        # sort by distance
        distances.sort(key=lambda x: x[0])

        # take k nearest
        k_nearest = distances[:self.k]

        # majority vote
        labels = [label for _, label in k_nearest]
        return Counter(labels).most_common(1)[0][0]

    def predict(self, X):
        return np.array([self._predict_single(x) for x in X])
