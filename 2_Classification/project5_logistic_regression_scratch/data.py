import numpy as np

def generate_data():
    np.random.seed(42)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = (X[:, 0] > 5).astype(int)  # pass if hours > 5
    return X, y
