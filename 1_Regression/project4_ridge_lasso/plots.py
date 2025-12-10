import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from data import load_data

X, y = load_data()
X_plot = np.linspace(-5, 5, 200).reshape(-1, 1)

for degree in [3, 10]:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)
    X_plot_poly = poly.transform(X_plot)

    lin = LinearRegression().fit(X_poly, y)
    ridge = Ridge(alpha=1.0).fit(X_poly, y)
    lasso = Lasso(alpha=0.1).fit(X_poly, y)

    plt.figure(figsize=(8,6))
    plt.scatter(X, y, s=15, color='black', label="Data")

    plt.plot(X_plot, lin.predict(X_plot_poly), label="Linear (no regularization)")
    plt.plot(X_plot, ridge.predict(X_plot_poly), label="Ridge")
    plt.plot(X_plot, lasso.predict(X_plot_poly), label="Lasso")

    plt.title(f"Degree {degree} — Effect of Regularization")
    plt.legend()
    plt.show()
