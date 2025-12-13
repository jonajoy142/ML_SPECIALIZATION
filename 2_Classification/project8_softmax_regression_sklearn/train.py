from sklearn.linear_model import LogisticRegression
from data import load_data
import joblib

X_train, X_test, y_train, y_test = load_data()

model = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=1000
)

model.fit(X_train, y_train)

joblib.dump(model, "softmax_model.joblib")

print("Model trained!")
print("Weights:\n", model.coef_)
print("Bias:\n", model.intercept_)
