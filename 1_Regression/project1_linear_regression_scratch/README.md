# Project 1 — Linear Regression From Scratch

A comprehensive implementation of Linear Regression from scratch using only NumPy, designed to build strong intuition for how machine learning algorithms work under the hood.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Concepts Covered](#concepts-covered)
- [Evaluation Metrics](#evaluation-metrics)
- [Example Results](#example-results)
- [FAQ](#faq)
- [Next Steps](#next-steps)

## 🎯 Overview

This project implements a complete Linear Regression model without using any machine learning libraries (such as scikit-learn). The implementation uses only NumPy for numerical computations, providing a deep understanding of:

- The mathematical foundations of linear regression
- Gradient descent optimization algorithm
- Model evaluation and performance metrics
- Synthetic data generation for learning purposes

## ✨ Features

- **Pure NumPy Implementation**: No external ML libraries required
- **Gradient Descent Optimization**: Custom implementation of the optimization algorithm
- **Comprehensive Evaluation**: Multiple metrics including MSE, RMSE, MAE, and R² Score
- **Visualization**: Cost curve and regression line plotting
- **Synthetic Data Generation**: Reproducible dataset generation for learning
- **Performance Metrics**: Inference time measurement

## 📦 Requirements

- Python 3.7+
- NumPy
- Matplotlib (for visualization)

## 📂 Project Structure

```
project1_linear_regression_scratch/
│
├── data.py               # Synthetic dataset generation
├── model.py              # Linear regression model implementation
├── train.py              # Model training script
├── eval.py               # Model evaluation and visualization
└── README.md             # Project documentation
```

## 🚀 Usage

### Training the Model

```bash
python train.py
```

This script will:
- Generate synthetic data
- Initialize the model parameters
- Train the model using gradient descent
- Display training progress

### Evaluating the Model

```bash
python eval.py
```

This script will:
- Re-train the model (currently re-trains from scratch)
- Calculate evaluation metrics
- Generate visualizations:
  - Cost (loss) curve
  - Learned regression line with data points

## 📚 Concepts Covered

### Linear Regression Basics

**Hypothesis Function:**
```
hθ(x) = θx + b
```

Where:
- **θ (theta)**: Slope parameter (weight)
- **b**: Bias parameter (intercept)
- **x**: Input feature(s)

**Key Concepts:**
- Understanding the relationship between features and target variables
- How `X.shape` determines data dimensions (samples × features)
- Parameter initialization and their role in predictions

### Gradient Descent

**Why Gradient Descent?**
Gradient descent is an optimization algorithm used to minimize the cost function by iteratively updating model parameters.

**Parameter Update Rules:**
```
θ := θ - α * dθ
b := b - α * db
```

Where:
- **α (alpha)**: Learning rate (controls step size)
- **dθ**: Gradient with respect to theta
- **db**: Gradient with respect to bias

**Key Concepts:**
- Role of learning rate in convergence
- Meaning of iterations/epochs
- Gradient computation from cost function derivatives

**Gradient Formulas:**
```python
d_theta = (1/n) * np.dot(X.T, (y_pred - y))
d_bias  = (1/n) * np.sum(y_pred - y)
```

These are the partial derivatives of the Mean Squared Error (MSE) cost function.

### Evaluation Metrics

The project implements and evaluates the following metrics:

- **MSE (Mean Squared Error)**: Average squared difference between predictions and actual values
- **RMSE (Root Mean Squared Error)**: Square root of MSE, in same units as target variable
- **MAE (Mean Absolute Error)**: Average absolute difference between predictions and actual values
- **R² Score**: Coefficient of determination, measures goodness of fit (0 to 1, higher is better)
- **Inference Time**: Time taken for prediction (typically microseconds for linear regression)

### Synthetic Data Generation

**Why Synthetic Data?**
- Provides controlled learning environment
- Allows understanding of how noise affects model performance
- Enables reproducibility for educational purposes

**Key Concepts:**
- `np.random.seed(42)`: Ensures reproducibility by fixing random number generation
- Noise addition: Simulates real-world imperfect data
- Impact of noise on learned parameters (e.g., bias shifting)

## 📊 Example Results

### Evaluation Metrics

```
--- Evaluation Metrics ---
MSE: 20.50
RMSE: 4.52
MAE: 3.57
R²: 0.9895
Inference Time: 0.0000047 seconds
```

### Interpretation

- **Slope (θ)**: ≈ 3.03
- **Bias (b)**: ≈ 8.42 (affected by noise in data)
- **R² Score**: ≈ 0.99 indicates excellent model fit
- **Inference Time**: Extremely fast (microseconds), demonstrating efficiency of linear regression

## ❓ FAQ

### 1. What is `np.random.seed(42)`?

`np.random.seed(42)` makes randomness repeatable by initializing the random number generator with a fixed seed value. This ensures:
- Same synthetic data generated every run
- Reproducible results for learning and debugging
- Without it, results would differ on each execution

### 2. What does `X.shape` represent?

`X.shape` shows the dimensions of the dataset:
- Format: `(samples, features)`
- Example: `(100, 1)` means 100 samples with 1 feature
- Essential for correct weight initialization and matrix operations

### 3. Why do we add noise in synthetic data?

Noise is added to:
- Simulate real-world imperfect data
- Make regression problems more realistic
- Understand why learned parameters may not exactly match true parameters
- Demonstrate model robustness

### 4. Why use `y_pred = dot(X, theta) + bias`?

This is the fundamental linear regression equation:
- Represents the hypothesis function `hθ(x) = θx + b`
- Initial predictions are usually incorrect
- Gradient descent iteratively adjusts θ and b to minimize error

### 5. Why these specific gradient formulas?

The gradients are derived from the partial derivatives of the MSE cost function:
```python
d_theta = (1/n) * np.dot(X.T, (y_pred - y))
d_bias  = (1/n) * np.sum(y_pred - y)
```

These formulas indicate the direction and magnitude of parameter adjustments needed to reduce prediction error.

### 6. Why does `eval.py` re-train the model instead of loading a saved model?

Currently, `eval.py` re-trains a new model using the same data. This is why results appear similar. Future enhancements will include:
- Model saving functionality
- Model loading capability
- Separate inference script for predictions

### 7. What is inference speed?

Inference speed measures the time taken for a single prediction:
- Linear regression is extremely fast (typically microseconds)
- Important for production applications requiring real-time predictions
- Measured using Python's `time` module

## 🌟 Next Steps

**Project 2 — Linear Regression using sklearn**

The next project will focus on:
- Comparing scratch implementation vs. scikit-learn
- Working with real-world datasets
- Measuring performance differences
- Validating correctness of the scratch implementation

---

**Note**: This project is part of a Machine Learning Specialization series designed to build foundational understanding through hands-on implementation.
