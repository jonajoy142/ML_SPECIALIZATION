import matplotlib.pyplot as plt
from data import load_data
from model import KNN

X_train, X_test, y_train, y_test = load_data()

model = KNN(k=3)
model.fit(X_train, y_train)

preds = model.predict(X_test)

plt.scatter(X_test[:,0], X_test[:,1], c=preds, cmap='viridis', s=50)
plt.title("KNN Classification (Scratch)")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()
