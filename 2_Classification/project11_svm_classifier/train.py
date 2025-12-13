from data import load_data
from sklearn.svm import SVC
import joblib

X_train, X_test, y_train, y_test = load_data()

model = SVC(kernel="linear", C=1.0)
model.fit(X_train, y_train)

joblib.dump(model, "svm_model.pkl")

print("SVM model trained and saved.")
print("Weights:", model.coef_)
print("Bias:", model.intercept_)
