import numpy as np
import matplotlib.pyplot as plt
from data import load_data
from model import SoftmaxRegression

X, y = load_data()

model = SoftmaxRegression(lr=0.05, iterations=2000)
model.fit(X, y)

# Create meshgrid
x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1
y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

grid = np.c_[xx.ravel(), yy.ravel()]
Z = model.predict(grid)
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.4)
plt.scatter(X[:,0], X[:,1], c=y, edgecolors='k')
plt.title("Softmax Regression Decision Boundaries")
plt.savefig("plot7.png")
plt.show()
