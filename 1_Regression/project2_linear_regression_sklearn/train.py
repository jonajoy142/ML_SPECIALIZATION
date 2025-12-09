from sklearn.linear_model import LinearRegression
from data import load_data

if __name__ == "__main__":
    X, y = load_data()

    model = LinearRegression()
    model.fit(X, y)

    print("Learned theta (coef):", model.coef_)
    print("Learned bias (intercept):", model.intercept_)
