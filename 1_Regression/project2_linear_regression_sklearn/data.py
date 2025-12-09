import numpy as np

def load_data():
    # Synthetic dataset: y = 3x + 10 + noise
    np.random.seed(42)
    X = np.linspace(0, 50, 100)
    noise = np.random.randn(100) * 5
    y = 3 * X + 10 + noise
    return X.reshape(-1, 1), y
