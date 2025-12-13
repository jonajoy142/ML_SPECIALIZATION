from sklearn.ensemble import RandomForestClassifier
from data import load_data
import joblib


X_train, X_test, y_train, y_test = load_data()

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)
model.fit(X_train, y_train)
joblib.dump(model, "random_forest.joblib")
print("Random Forest trained and saved.")
