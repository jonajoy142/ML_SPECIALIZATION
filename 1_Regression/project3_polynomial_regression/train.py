from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from data import load_data

if __name__ == "__main__":
    X, y = load_data()

    for degree in [1, 2, 3, 10]:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_poly = poly.fit_transform(X)

        model = LinearRegression()
        model.fit(X_poly, y)

        print(f"\nDegree {degree} model trained.")
        print("Coefficients:", model.coef_)
        print("Intercept:", model.intercept_)
