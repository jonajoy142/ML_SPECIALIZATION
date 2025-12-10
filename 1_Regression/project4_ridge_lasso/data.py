import numpy as np

def load_data():
    np.random.seed(42)
    X = np.linspace(-5, 5, 100)
    noise = np.random.normal(0, 10, size=100)
    
    # True cubic relationship
    y = 0.5*X**3 - 2*X**2 + 3*X + 5 + noise
    
    return X.reshape(-1, 1), y
