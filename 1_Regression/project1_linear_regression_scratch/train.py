# train.py
import matplotlib.pyplot as plt
from data import load_data
from model import LinearRegressionScratch

# Load dataset
X, y = load_data()

# Train the model
model = LinearRegressionScratch(learning_rate=0.001, iterations=40000)
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

print("Learned theta:", model.theta)
print("Learned bias:", model.bias)

# Plot results
plt.scatter(X, y, color="blue", label="Data points")
plt.plot(X, y_pred, color="red", label="Predicted line")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()
