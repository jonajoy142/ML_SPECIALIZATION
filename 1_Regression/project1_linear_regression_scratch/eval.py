import time
import numpy as np
import matplotlib.pyplot as plt
from data import load_data
from model import LinearRegressionScratch


# Evaluation Metrics

def mean_squared_error(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


# Main Evaluation Script

if __name__ == "__main__":

    # Load data
    X, y = load_data()

    # Train model
    model = LinearRegressionScratch(learning_rate=0.001, iterations=10000)
    model.fit(X, y)

    # Prediction + Inference Time
    start = time.time()
    y_pred = model.predict(X)
    end = time.time()
    inference_time = end - start

    # Print learned parameters
    print("Learned theta:", model.theta)
    print("Learned bias:", model.bias)

    # Compute Metrics
    mse  = mean_squared_error(y, y_pred)
    rmse = root_mean_squared_error(y, y_pred)
    mae  = mean_absolute_error(y, y_pred)
    r2   = r2_score(y, y_pred)

    # Print metrics
    print("\n--- Evaluation Metrics ---")
    print(f"MSE:  {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE:  {mae:.4f}")
    print(f"R²:   {r2:.4f}")
    print(f"Inference Time: {inference_time:.10f} seconds")

    
    # Plot Cost Over Iterations
    
    plt.figure(figsize=(7, 5))
    plt.plot(model.cost_history, label="Cost (MSE)")
    plt.xlabel("Iteration")
    plt.ylabel("Cost")
    plt.title("Cost Decrease Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

    
    # Plot Data and Learned Line
    
    plt.figure(figsize=(7, 5))
    plt.scatter(X, y, color="blue", alpha=0.6, label="Data Points")
    plt.plot(X, y_pred, color="red", linewidth=2, label="Learned Line")
    plt.xlabel("X")
    plt.ylabel("y")
    plt.title("Linear Regression Fit")
    plt.legend()
    plt.grid(True)
    plt.show()
