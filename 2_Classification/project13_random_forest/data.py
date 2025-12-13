# data.py
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def load_data():
    X, y = make_classification(
        n_samples=300,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_classes=2,
        random_state=42
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)
