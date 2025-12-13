from data import load_data
from model import SoftmaxRegression
from sklearn.model_selection import train_test_split
import numpy as np

X, y = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = SoftmaxRegression(lr=0.05, iterations=2000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = np.mean(y_pred == y_test)

print("Accuracy:", round(acc, 4))
