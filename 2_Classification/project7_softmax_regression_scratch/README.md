# Project 7 — Softmax Regression From Scratch (Multiclass Classification)

A comprehensive implementation of Softmax Regression from scratch using only NumPy, designed to understand multiclass classification and how it extends binary logistic regression.

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

Softmax Regression is the multiclass version of Logistic Regression. While logistic regression is used when there are only 2 classes, Softmax regression is used when there are 3 or more classes.

This project implements a complete Softmax Regression model without using any machine learning libraries (such as scikit-learn). The implementation uses only NumPy for numerical computations, providing a deep understanding of:

- The mathematical foundations of multiclass classification
- Softmax function and probability distributions
- One-hot encoding for multiclass labels
- Cross-entropy loss for multiclass problems
- Gradient descent optimization for multiclass classification

## ✨ Features

- **Pure NumPy Implementation**: No external ML libraries required
- **Softmax Function**: Custom implementation of the softmax activation function
- **One-Hot Encoding**: Automatic conversion of class labels to one-hot vectors
- **Gradient Descent Optimization**: Custom implementation for multiclass classification
- **Decision Boundary Visualization**: Plotting of classification boundaries
- **Synthetic Data Generation**: Using `make_blobs` for clean multiclass datasets

## 📦 Requirements

- Python 3.7+
- NumPy
- Matplotlib (for visualization)
- scikit-learn (for `make_blobs` dataset generation only)

## 📂 Project Structure

```
project7_softmax_regression_scratch/
│
├── data.py               # Synthetic dataset generation using make_blobs
├── model.py              # Softmax regression model implementation
├── train.py              # Model training script
├── eval.py               # Model evaluation and accuracy calculation
├── plot.py               # Decision boundary visualization
├── plot_7.png            # Generated visualization
└── README.md             # Project documentation
```

## 🚀 Usage

### Training the Model

```bash
python train.py
```

This script will:
- Generate synthetic multiclass data using `make_blobs`
- Initialize the model parameters
- Train the model using gradient descent
- Display training progress and learned weights/bias

### Evaluating the Model

```bash
python eval.py
```

This script will:
- Load the trained model
- Calculate evaluation metrics (accuracy)
- Display performance results

### Visualizing Decision Boundaries

```bash
python plot.py
```

This script will:
- Generate decision boundary plots
- Visualize data points and classification regions
- Save the plot as `plot_7.png`

## 📚 Concepts Covered

### Softmax Regression Basics

**What is Softmax Regression?**

Imagine we want to classify points into 3 groups: Red, Blue, and Green. Unlike logistic regression (which says yes/no), Softmax Regression outputs probabilities for each class:

| Class | Probability |
|-------|-------------|
| 0 (Red) | 0.02 |
| 1 (Blue) | 0.75 |
| 2 (Green) | 0.23 |

**Predicted class** = the one with highest probability → Blue (0.75)

**Key Differences:**
- **Logistic Regression**: Binary classification (2 classes), uses sigmoid
- **Softmax Regression**: Multiclass classification (3+ classes), uses softmax

### Softmax Function

**Intuition:**

The softmax function converts raw output values (logits) into probabilities. Suppose the model outputs raw values:

```
[3, 1, 0]
```

These numbers don't look like probabilities. Softmax does two things:
1. Make all numbers positive using exponential
2. Make them add up to 1 by dividing by the total

**Softmax Formula:**

```python
exp = np.exp(z - np.max(z, axis=1, keepdims=True))
return exp / np.sum(exp, axis=1, keepdims=True)
```

**Meaning:**
- Converts ANY numbers into probabilities
- Bigger numbers → bigger probabilities
- All probabilities add up to 1.0

The subtraction of `np.max(z)` is for numerical stability to prevent overflow.

### Why Use make_blobs?

**Before (previous projects):**
- Straight line data
- Cubic curve data
- Binary classification using simple rules

**Now (multiclass):**
We need multiple clusters, one per class. `make_blobs` creates nice, clean 2D clusters like this:

```
   ●●●  Class 0
          ●●● Class 1
     ●●● Class 2
```

This makes multiclass learning very intuitive and allows us to clearly visualize decision boundaries.

### One-Hot Encoding

**What is One-Hot Encoding?**

One-hot encoding converts class labels into binary vectors. Suppose the true classes are:

```python
y = [1, 0, 2]
```

We turn them into:

- Class 0 → [1, 0, 0]
- Class 1 → [0, 1, 0]
- Class 2 → [0, 0, 1]

**Why?**

Because softmax outputs 3 probabilities (one for each class), so the labels must also be 3 values. This format allows us to compare predicted probability distributions with true class distributions using cross-entropy loss.

### Model Training

**Training Process:**

1. **Start with random weights**: Initialize weight matrix W and bias vector b
2. **Predict probabilities**: Compute logits and apply softmax
   ```python
   logits = np.dot(X, self.W) + self.b
   y_pred = softmax(logits)
   ```
3. **Compare with true one-hot labels**: Calculate cross-entropy loss
4. **Adjust weights**: Use gradient descent to update parameters
5. **Repeat**: Iterate until convergence

**Gradient Computation:**

The gradients for softmax regression with cross-entropy loss:

```python
dW = (1/m) * np.dot(X.T, (y_pred - Y))
db = (1/m) * np.sum((y_pred - Y), axis=0, keepdims=True)
```

Where:
- `X` is the input features
- `y_pred` is the predicted probability distribution
- `Y` is the true one-hot encoded labels
- `m` is the number of samples

### Prediction

**How Predictions Work:**

After training, to predict a class:

```python
logits = np.dot(X, self.W) + self.b
y_pred = softmax(logits)
return np.argmax(y_pred, axis=1)
```

- Softmax gives probabilities for each class
- `argmax` picks the class with highest probability
- Returns the predicted class index

## 📊 Evaluation Metrics

### Accuracy

Unlike binary classification (which uses precision, recall, F1-score), multiclass classification often uses **accuracy** as the primary metric, especially when:

- The dataset is clean
- Classes are balanced
- There's no real-world class imbalance problem

**Accuracy Formula:**

```
Accuracy = (Number of correct predictions) / (Total number of predictions)
```

**Interpretation:**
- Accuracy = 0.95 means the model correctly classified 95 out of 100 points
- Accuracy = 1.0 means perfect classification (all points correctly classified)

### Why Only Accuracy?

For this project, accuracy is sufficient because:
- The synthetic dataset is balanced
- Classes are well-separated
- No class imbalance issues

In real-world applications with imbalanced classes, additional metrics like precision, recall, and F1-score per class would be more informative.

## 📊 Example Results

### Training Output

```
Training complete!
Weights:
 [[-0.65946386  1.29511285 -0.63564899]
 [ 0.60437807  0.08425852 -0.68863659]]
Bias:
 [[-0.25067116  0.28737319 -0.03670203]]
```

### Evaluation Output

```
Accuracy: 1.0
```

### Visualization

The project generates decision boundary plots showing:
- Data points colored by their true class
- Decision boundaries separating different classes
- Smooth classification regions

![Softmax Regression Decision Boundaries](./plot_7.png)

### Interpretation

- **Perfect Accuracy (1.0)**: The model correctly classifies all test points
- **Smooth Decision Boundaries**: Indicates the model learned well-separated regions
- **Weight Matrix**: Each column corresponds to one class, learned through gradient descent

## ❓ FAQ

### 1. What is Softmax? Why not Sigmoid?

- **Sigmoid**: Gives ONE probability (binary: 0 or 1), used for binary classification
- **Softmax**: Gives MANY probabilities (one per class), used for multiclass classification

Both convert logits to probabilities, but sigmoid handles 2 classes while softmax handles 3+ classes.

### 2. Why use make_blobs?

Multiclass classification needs multiple clusters. `make_blobs` creates 3+ clearly separated groups, making it easy to:
- Understand how the algorithm works
- Visualize decision boundaries
- Verify the model learned correctly

### 3. What does the softmax code do?

```python
exp = np.exp(z - np.max(z, axis=1, keepdims=True))
exp / np.sum(exp, axis=1, keepdims=True)
```

- `np.exp(z - np.max(z))`: Makes all values positive (using exponential) and prevents overflow
- `exp / np.sum(exp)`: Normalizes them to sum to 1
- Turns raw scores → probabilities

### 4. Why one-hot encode the labels?

Because softmax predicts a probability vector (one probability per class), not a single number. To compare predictions with true labels using cross-entropy loss, we need the true labels in the same format (one-hot vectors).

### 5. Why only accuracy for evaluation?

Because this dataset is balanced, clean, and simple. No need for precision/recall unless data is imbalanced. In real-world scenarios with imbalanced classes, you would use additional metrics.

### 6. How does gradient descent work for multiclass?

The gradient descent algorithm:
1. Computes predictions using current weights
2. Calculates the difference between predictions and true labels
3. Updates weights proportionally to reduce this difference
4. Repeats until convergence

The gradients `dW` and `db` tell us how to adjust weights to minimize the cross-entropy loss.

### 7. What does "high accuracy" + "smooth decision boundaries" mean?

- **High accuracy**: The model correctly classifies most/all test points
- **Smooth decision boundaries**: The model learned well-separated regions without overfitting
- Together, these indicate the model generalizes well and learned the underlying patterns correctly

## 🌍 Real-World Examples of Softmax Regression

Softmax regression is used whenever you have more than 2 categories:

1. **Digit Recognition (0–9)**: Classifying handwritten digits
2. **Product Category Prediction**: Categorizing products in e-commerce
3. **Disease Classification**: Classifying medical conditions
4. **Image Classification**: Classifying images (dog, cat, panda, etc.)
5. **Sentiment Analysis**: Classifying text sentiment (positive, neutral, negative)

## 🌟 Next Steps

**Project 8 — Softmax Regression using sklearn**

The next project will focus on:
- Implementing multiclass classification using scikit-learn's `LogisticRegression`
- Comparing scratch implementation vs. scikit-learn
- Understanding `multi_class="multinomial"` parameter
- Working with additional evaluation metrics (precision, recall, F1-score)

**Foundation for Advanced Topics:**

Softmax Regression is the foundation for:
- Neural Networks (output layer often uses softmax)
- CNNs (Convolutional Neural Networks)
- RNNs (Recurrent Neural Networks)
- Deep Learning classification tasks

---

**Note**: This project is part of a Machine Learning Specialization series designed to build foundational understanding through hands-on implementation.
