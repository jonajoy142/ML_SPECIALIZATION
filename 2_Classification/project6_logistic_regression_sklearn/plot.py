import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from data import generate_data

X, y = generate_data()
model = LogisticRegression()
model.fit(X, y)

# Decision boundary: θx + b = 0  →  x = -b/θ
theta = model.coef_[0][0]
bias = model.intercept_[0]
decision_boundary = -bias / theta

plt.scatter(X[y==0], y[y==0], color="red", label="Class 0")
plt.scatter(X[y==1], y[y==1], color="blue", label="Class 1")

plt.axvline(decision_boundary, color="green", linestyle="--",
            label=f"Decision Boundary (x={decision_boundary:.2f})")

plt.title("Logistic Regression Decision Boundary")
plt.xlabel("Feature X")
plt.ylabel("Class")
plt.legend()
plt.grid(True)
plt.savefig("plot6.png")
plt.show()
