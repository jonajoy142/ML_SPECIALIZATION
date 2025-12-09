import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from data import load_data

if __name__ == "__main__":
    X, y = load_data()
    X_plot = np.linspace(-5, 5, 500).reshape(-1, 1)

    plt.scatter(X, y, color="blue", label="Data", alpha=0.4)

    for degree in [1, 2, 3, 10]:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)
        X_plot_poly = poly.transform(X_plot)

        model = LinearRegression()
        model.fit(X_poly, y)
        y_plot = model.predict(X_plot_poly)

        plt.plot(X_plot, y_plot, label=f"Degree {degree}", linewidth=2)

    plt.title("Polynomial Regression Fits (Underfit → Good Fit → Overfit)")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()
