import numpy as np
import matplotlib.pyplot as plt
import joblib
from data import load_data

model = joblib.load("xgboost.joblib")
X_train, X_test, y_train, y_test = load_data()

x_min, x_max = X_train[:,0].min()-1, X_train[:,0].max()+1
y_min, y_max = X_train[:,1].min()-1, X_train[:,1].max()+1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X_train[:,0], X_train[:,1], c=y_train)
plt.title("XGBoost Decision Boundary")
plt.savefig("xgboost.png")
plt.show()
