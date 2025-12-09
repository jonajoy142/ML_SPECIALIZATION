# Project 2 — Linear Regression using scikit-learn

A practical implementation of Linear Regression using scikit-learn, comparing it with the scratch implementation from Project 1 to understand the differences between analytical solutions and iterative optimization methods.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Concepts Covered](#concepts-covered)
- [Comparison with Project 1](#comparison-with-project-1)
- [Evaluation Metrics](#evaluation-metrics)
- [Why Project 2 Matters](#why-project-2-matters)
- [Next Steps](#next-steps)

## 🎯 Overview

This project implements Linear Regression using scikit-learn's `LinearRegression` class and compares it with the scratch implementation from Project 1. The goal is to understand:

- How sklearn solves linear regression analytically using Ordinary Least Squares (OLS)
- How analytical solutions compare to gradient descent optimization
- How to evaluate regression performance using standard metrics
- Why sklearn's training is instantaneous compared to iterative methods

## ✨ Features

- **scikit-learn Implementation**: Using `LinearRegression` for production-ready ML
- **Analytical Solution**: Closed-form OLS solution instead of iterative optimization
- **Performance Comparison**: Comparing scratch vs. sklearn implementations
- **Comprehensive Evaluation**: Standard regression metrics (MSE, RMSE, MAE, R²)
- **Inference Speed**: Measuring prediction time
- **Model Validation**: Verifying correctness against scratch implementation

## 📦 Requirements

- Python 3.7+
- NumPy
- scikit-learn
- Matplotlib (optional, for visualization)

## 📂 Project Structure

```
project2_linear_regression_sklearn/
│
├── data.py          # Synthetic dataset generation
├── train.py         # Model training using sklearn
├── eval.py          # Model evaluation and metrics
└── README.md        # Project documentation
```

## 🚀 Usage

### Training the Model

```bash
python train.py
```

This script will:
- Load synthetic data
- Initialize sklearn's `LinearRegression` model
- Fit the model to the data
- Display learned parameters (coefficients and intercept)

**Output:**
```
Learned theta (coef): [3.03]
Learned bias (intercept): 8.42
```

### Evaluating the Model

```bash
python eval.py
```

This script will:
- Train the sklearn model
- Generate predictions
- Calculate evaluation metrics
- Display performance results

**Example Output:**
```
--- Evaluation Metrics (sklearn) ---
MSE:  20.5000
RMSE: 4.5277
MAE:  3.5700
R²:   0.9895
Inference Time: 0.0000047000 seconds
```

## 📚 Concepts Covered

### Using LinearRegression from sklearn

**Key Features:**
- Fits a linear model using **Ordinary Least Squares (OLS)**
- No need to manually implement gradient descent
- Extremely fast and numerically stable
- Production-ready implementation

**Basic Usage:**
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)
```

**Model Attributes:**
- `model.coef_`: Learned slope parameter (θ)
- `model.intercept_`: Learned bias parameter (b)

### Analytical Solution vs. Gradient Descent

**Project 1 (Scratch Implementation):**
- Uses iterative **gradient descent** optimization
- Updates parameters step-by-step until convergence
- Requires learning rate tuning
- Multiple iterations needed

**Project 2 (sklearn Implementation):**
- Uses **closed-form analytical solution** (OLS)
- Computes optimal parameters directly using matrix operations
- No iterations required
- Instantaneous training

**Mathematical Comparison:**

| Method | Approach | Speed | Complexity |
|--------|----------|-------|------------|
| Scratch (Gradient Descent) | Iterative optimization | Slower (requires iterations) | Manual implementation |
| sklearn (OLS) | Analytical solution | Instant | Library handles complexity |

**Key Insight:**
Both methods should produce nearly identical parameters (within numerical precision), validating the correctness of the scratch implementation.

### Model Inference using sklearn

**Prediction Method:**
```python
y_pred = model.predict(X)
```

**Inference Speed:**
- sklearn's implementation is highly optimized
- Extremely fast prediction time (microseconds)
- Comparable to scratch implementation for simple cases
- Significant advantage for large datasets

### Computing Evaluation Metrics

The project uses scikit-learn's built-in metrics:

- **MSE (Mean Squared Error)**: `mean_squared_error(y, y_pred)`
- **RMSE (Root Mean Squared Error)**: `np.sqrt(mse)`
- **MAE (Mean Absolute Error)**: `mean_absolute_error(y, y_pred)`
- **R² Score**: `r2_score(y, y_pred)` - measures variance explained
- **Inference Time**: Time taken for prediction

### Understanding Regression Behavior

**Parameter Learning:**
- **Slope (θ)**: Learned accurately, typically ≈ 3.0
- **Bias (intercept)**: May shift slightly (≈ 8-10) due to noise in data
- Small variations between methods due to numerical precision and randomness

**Fitted Lines:**
- Both scratch and sklearn implementations should produce similar regression lines
- Minor differences may occur due to:
  - Random initialization in gradient descent
  - Numerical precision differences
  - Convergence criteria

## 🔍 Comparison with Project 1

### Expected Parameters

Both Project 1 (scratch) and Project 2 (sklearn) should learn approximately:

- **θ (slope)**: ≈ 3.0
- **bias/intercept**: ≈ 8-10
  - Small variations occur due to noise in synthetic data

### Why sklearn is Faster

**Project 1 (Gradient Descent):**
- Iterative process: "Try until you reach the best answer"
- Requires multiple iterations to converge
- Each iteration updates parameters slightly
- Training time: O(iterations × samples)

**Project 2 (sklearn OLS):**
- Analytical solution: "Jump directly to the best answer using math"
- Computes optimal parameters in one step
- Uses matrix operations: θ = (X^T X)^(-1) X^T y
- Training time: O(samples × features²)

**Visual Comparison:**

| Aspect | Scratch Model | sklearn |
|--------|---------------|---------|
| Training Method | Gradient Descent | OLS (Analytical) |
| Training Speed | Slower (iterative) | Instant |
| Code Complexity | Higher (manual implementation) | Lower (library) |
| Accuracy | Same (within precision) | Same (within precision) |
| Use Case | Learning/Education | Production |

## 📊 Evaluation Metrics

### Metrics Explained

- **MSE (Mean Squared Error)**: Average of squared differences between predictions and actual values
- **RMSE (Root Mean Squared Error)**: Square root of MSE, in same units as target variable
- **MAE (Mean Absolute Error)**: Average absolute difference, less sensitive to outliers than MSE
- **R² Score**: Coefficient of determination (0 to 1), higher is better
- **Inference Time**: Time for single prediction, typically microseconds

### Typical Results

For the synthetic dataset (y = 3x + 10 + noise):
- **MSE**: ~20-25
- **RMSE**: ~4.5-5.0
- **MAE**: ~3.5-4.0
- **R²**: ~0.98-0.99 (excellent fit)
- **Inference Time**: < 0.00001 seconds

## 🎓 Why Project 2 Matters

Project 2 demonstrates:

1. **How Real ML Libraries Work**
   - Understanding production-ready implementations
   - Learning to use industry-standard tools

2. **Validating Scratch Implementation**
   - Comparing results validates correctness of Project 1
   - Builds confidence in understanding fundamentals

3. **Performance Comparison**
   - Training speed differences
   - Accuracy validation
   - When to use each approach

4. **Proper Model Evaluation**
   - Standard metrics and their interpretation
   - Best practices for regression evaluation

5. **Power of Closed-Form Solutions**
   - Understanding when analytical solutions exist
   - Appreciating optimization trade-offs

### Foundation for Future Learning

This project builds the foundation needed for:
- **Polynomial Regression**: Extending linear models to non-linear relationships
- **Regularization**: L1/L2 regularization techniques
- **Neural Networks**: Understanding optimization in complex models
- **All Future ML Algorithms**: Core concepts transfer across domains

## 🌟 Next Steps

### Project 3 — Polynomial Regression

In the next project, you will learn:

#### ⭐ Modeling Curved Relationships

Linear regression fits a straight line, but many real-world problems are non-linear:

- **Hours studied vs exam score**: Diminishing returns
- **Temperature vs electricity usage**: Seasonal patterns
- **Age vs medical risk score**: Non-linear health trends

Polynomial regression allows the model to learn curves using:
- Features: `x, x², x³, ...`
- Capturing non-linear patterns in data

#### ⭐ Understanding Underfitting & Overfitting

You will visualize and understand:

- **Underfitting**: Model too simple (straight line on curved data)
  - High training error
  - High test error
  - Model cannot capture patterns

- **Overfitting**: Model too complex (wiggly line that memorizes noise)
  - Low training error
  - High test error
  - Model memorizes training data

**Learning Objectives:**
- Plotting training vs. test error
- Comparing different polynomial degrees
- Selecting optimal model complexity
- Understanding bias-variance trade-off

---

**Note**: This project is part of a Machine Learning Specialization series designed to build foundational understanding through hands-on implementation and comparison of different approaches.
