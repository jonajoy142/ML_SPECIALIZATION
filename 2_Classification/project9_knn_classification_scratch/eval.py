from data import load_data
from model import KNN
import numpy as np

X_train, X_test, y_train, y_test = load_data()

model = KNN(k=3)
model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = np.mean(predictions == y_test)
print(f"Accuracy: {accuracy:.2f}")
