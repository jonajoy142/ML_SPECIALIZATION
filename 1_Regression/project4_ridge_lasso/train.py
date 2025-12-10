from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge, Lasso
from data import load_data

X, y = load_data()

for degree in [1, 3, 10]:
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly = poly.fit_transform(X)

    print(f"\n--- Degree {degree} ---")

    ridge = Ridge(alpha=1.0)
    ridge.fit(X_poly, y)
    print("Ridge Coeff:", ridge.coef_)

    lasso = Lasso(alpha=0.1)
    lasso.fit(X_poly, y)
    print("Lasso Coeff:", lasso.coef_)
