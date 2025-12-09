import time
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from data import load_data

if __name__ == "__main__":
    X, y = load_data()

    # Train model
    model = LinearRegression()
    model.fit(X, y)

    # Predictions + inference time
    start = time.time()
    y_pred = model.predict(X)
    end = time.time()

    inference_time = end - start

    # Metrics
    mse  = mean_squared_error(y, y_pred)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(y, y_pred)
    r2   = r2_score(y, y_pred)

    print("\n--- Evaluation Metrics (sklearn) ---")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
    print(f"Inference Time: {inference_time:.10f} seconds")
