from data import load_data
from model import SoftmaxRegression

X, y = load_data()

model = SoftmaxRegression(lr=0.05, iterations=2000)
model.fit(X, y)

print("Training complete!")
print("Weights:\n", model.W)
print("Bias:\n", model.b)
