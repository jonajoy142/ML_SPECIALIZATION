#We will create non-linear synthetic data:
#y=0.5x^3−2x^2+3x+5+noise


import numpy as np

def load_data():
    np.random.seed(42)
    X = np.linspace(-5, 5, 100)
    noise = np.random.randn(100) * 10
    y = 0.5 * X**3 - 2 * X**2 + 3 * X + 5 + noise
    return X.reshape(-1, 1), y

