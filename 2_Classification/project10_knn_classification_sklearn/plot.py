import numpy as np
import matplotlib.pyplot as plt
from data import load_data
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = load_data()

model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Create mesh
x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 200),
    np.linspace(y_min, y_max, 200)
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, edgecolor="k")
plt.title("KNN Decision Boundary (sklearn)")
plt.show()
