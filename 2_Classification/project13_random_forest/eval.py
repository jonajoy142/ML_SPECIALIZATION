from sklearn.metrics import accuracy_score, classification_report
import joblib
from data import load_data


model = joblib.load("random_forest.joblib")
X_train, X_test, y_train, y_test = load_data()
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))
