from data import load_data
from sklearn.metrics import accuracy_score
import joblib

X_train, X_test, y_train, y_test = load_data()
model = joblib.load("svm_model.pkl")

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

print("Accuracy:", acc)
