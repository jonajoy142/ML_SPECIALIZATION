import numpy as np
import matplotlib.pyplot as plt
from model import LogisticRegressionScratch
from data import generate_data

# Load synthetic dataset
X, y = generate_data()

# Train the model
model = LogisticRegressionScratch(lr=0.01, iterations=5000)
model.fit(X, y)

# For decision boundary, we sort values for a smooth line
x_line = np.linspace(min(X[:,0]) - 1, max(X[:,0]) + 1, 100)

# Decision boundary formula:
# sigmoid(θx + b) = 0.5  →  θx + b = 0  →  x = -b / θ
theta = model.theta[0]
bias = model.bias

boundary = -(bias / theta)

# PREDICTION POINTS
y_pred = model.predict(X)

# PLOT
plt.figure(figsize=(8, 6))

# Plot class 0
plt.scatter(X[y==0], np.zeros_like(X[y==0]), color="red", label="Class 0")

# Plot class 1
plt.scatter(X[y==1], np.ones_like(X[y==1]), color="blue", label="Class 1")

# Decision boundary line
plt.axvline(boundary, color="green", linestyle="--", label=f"Decision Boundary (x={boundary:.2f})")

plt.title("Logistic Regression — Decision Boundary")
plt.xlabel("Feature X")
plt.yticks([0, 1])
plt.legend()
plt.grid(True)
plt.show()
