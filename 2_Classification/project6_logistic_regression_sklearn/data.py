# We will use the same data from Project 5, but written cleanly.
import numpy as np

def generate_data():
    np.random.seed(42)

    X = np.linspace(0, 10, 200).reshape(-1, 1)
    y = (X[:, 0] > 5).astype(int)

    return X, y
