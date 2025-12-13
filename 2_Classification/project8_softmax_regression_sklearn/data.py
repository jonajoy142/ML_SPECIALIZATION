import numpy as np
from sklearn.datasets import make_blobs

def load_data():
    X, y = make_blobs(
        n_samples=300,
        centers=3,
        cluster_std=1.2,
        random_state=42
    )
    return X, y
