from sklearn.linear_model import LogisticRegression
from data import generate_data

X, y = generate_data()

model = LogisticRegression()
model.fit(X, y)

print("Model trained!")
print("Coefficient:", model.coef_)
print("Bias:", model.intercept_)
