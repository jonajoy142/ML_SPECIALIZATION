import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from data import load_data

_, X_test, _, y_test = load_data()

model = joblib.load("softmax_model.joblib")

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred, average="macro"))
print("Recall:", recall_score(y_test, y_pred, average="macro"))
print("F1 Score:", f1_score(y_test, y_pred, average="macro"))
