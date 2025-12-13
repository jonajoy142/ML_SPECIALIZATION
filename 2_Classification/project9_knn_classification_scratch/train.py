from data import load_data
from model import KNN

X_train, X_test, y_train, y_test = load_data()

model = KNN(k=3)
model.fit(X_train, y_train)

print("KNN model trained with k=3")
