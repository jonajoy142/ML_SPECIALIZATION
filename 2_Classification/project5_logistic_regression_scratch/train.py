from data import generate_data
from model import LogisticRegressionScratch

X, y = generate_data()

model = LogisticRegressionScratch(lr=0.1, iterations=2000)
model.fit(X, y)

print("Theta:", model.theta)
print("Bias:", model.bias)
