# train.py
from sklearn.tree import DecisionTreeClassifier
from data import load_data
import joblib

X_train, X_test, y_train, y_test = load_data()

model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)

joblib.dump(model, "decision_tree_model.joblib")

print("Decision Tree trained and saved.")
print("Tree depth:", model.get_depth())
print("Number of leaves:", model.get_n_leaves())
