from data import generate_data
from model import LogisticRegressionScratch
from sklearn.metrics import accuracy_score

X, y = generate_data()
model = LogisticRegressionScratch(lr=0.1, iterations=2000)
model.fit(X, y)

y_pred = model.predict(X)
print("Accuracy:", accuracy_score(y, y_pred))
