import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

def load_data():
    X, y = make_blobs(
        n_samples=300,
        centers=3,
        cluster_std=1.5,
        random_state=42
    )

    return train_test_split(X, y, test_size=0.2, random_state=42)
