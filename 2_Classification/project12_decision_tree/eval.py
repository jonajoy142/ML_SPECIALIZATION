# eval.py
import joblib
from data import load_data
from sklearn.metrics import accuracy_score, classification_report

X_train, X_test, y_train, y_test = load_data()

model = joblib.load("decision_tree_model.joblib")

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
