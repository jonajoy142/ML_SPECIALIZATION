from xgboost import XGBClassifier
from data import load_data
import joblib

X_train, X_test, y_train, y_test = load_data()

model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    use_label_encoder=False,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

joblib.dump(model, "xgboost.joblib")
print("XGBoost model trained and saved.")
