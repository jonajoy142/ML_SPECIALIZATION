# Project 6 — Logistic Regression using scikit-learn

This project implements binary classification using logistic regression from `sklearn` and evaluates the model using **Accuracy, Precision, Recall, and F1 Score**. It follows after:

- **Project 5** → Logistic Regression from scratch  
- **Project 6** → Logistic Regression using sklearn

---

## 📋 Table of Contents

- [Overview](#overview)
- [What Logistic Regression Does](#what-logistic-regression-does)
- [How Logistic Regression Works (Simple)](#how-logistic-regression-works-simple)
- [Why Logistic Regression ≠ Linear Regression](#why-logistic-regression--linear-regression)
- [Dataset Used (Synthetic)](#dataset-used-synthetic)
- [Model Training (sklearn)](#model-training-sklearn)
- [Evaluation Metrics](#evaluation-metrics)
- [Your Evaluation Results (Validated)](#your-evaluation-results-validated)
- [Decision Boundary Plot](#decision-boundary-plot)
- [Full Summary](#full-summary)
- [Real-World Example](#real-world-example)
- [Next Project](#next-project)

---

## Overview

Binary classification with scikit-learn’s `LogisticRegression`, focusing on probability-based decisions and full evaluation (Accuracy, Precision, Recall, F1).

---

## 🧠 1. What Logistic Regression Does

Logistic Regression predicts probabilities for two classes:

- Class 0 (negative)  
- Class 1 (positive)

It is used when output is categorical:

| Example           | Output            |
|-------------------|-------------------|
| Spam Detection    | spam / not spam   |
| Tumor Diagnosis   | malignant / benign|
| Credit Default    | yes / no          |
| Exam pass         | pass / fail       |

---

## ⚙️ 2. How Logistic Regression Works (Simple)

It starts like Linear Regression:

```
z = θ·x + b
```

But instead of predicting a number, it feeds `z` into the sigmoid function:

**Sigmoid (key concept)**

[
\sigma(z) = \frac{1}{1 + e^{-z}}
]

Produces a value between 0 and 1 → interprets as probability of class 1.

**Decision rule:**

```
σ(z) ≥ 0.5 ⇒ predict 1
σ(z) < 0.5 ⇒ predict 0
```

---

## 🧠 3. Why Logistic Regression ≠ Linear Regression

| Linear Regression | Logistic Regression |
|-------------------|---------------------|
| Predicts continuous values (−∞ to +∞) | Predicts probabilities (0–1) |
| Uses MSE loss | Uses Binary Cross-Entropy loss |
| Fits a straight line | Fits a sigmoid S-curve |
| Not suitable for classification | Made specifically for classification |

---

## 🧪 4. Dataset Used (Synthetic)

We create a simple classification dataset:

```python
X = np.linspace(0, 10, 200).reshape(-1, 1)
y = (X[:, 0] > 5).astype(int)
```

Meaning:

- If feature value > 5 → class 1  
- Else class 0

This gives a perfectly separable dataset, so the model should reach nearly 100% performance.

---

## 📈 5. Model Training (sklearn)

```python
model = LogisticRegression()
model.fit(X, y)
```

After training, you got:

- **Coefficient:** `[[3.19781609]]`  
- **Bias:** `[-15.98901022]`

Meaning:

- Positive coefficient → probability increases as x increases  
- Negative bias → shifts decision boundary

Decision boundary occurs when:

```
θx + b = 0  ⇒  x = −b/θ
x = −(−15.989) / 3.197 ≈ 5.0
```

Perfect — model learned exactly what we expect.

---

## 🎯 6. Evaluation Metrics

main questions:

“What are Precision, Recall, F1 Score and why weren’t they used in scratch implementation?”

✔ **Accuracy**

```
Accuracy = Correct predictions / Total samples
```

Useful when dataset is balanced.

✔ **Precision**

Of all predicted positives, how many were correct?

```
Precision = TP / (TP + FP)
```

✔ **Recall**

Of all actual positives, how many did we detect?

```
Recall = TP / (TP + FN)
```

✔ **F1 Score**

Balanced mean of Precision and Recall.

```
F1 = 2 * Precision * Recall / (Precision + Recall)
```

**Why these were NOT in Project 5 (scratch)?**

Because:

- Scratch projects focus on core math  
- Accuracy was enough for understanding gradient descent  
- Computing F1 manually would distract from logistic regression theory  
- Sklearn makes these metrics easy → so we use them here.

---

## 📊 7. Your Evaluation Results (VALIDATED)

- Accuracy: **1.0**  
- Precision: **1.0**  
- Recall: **1.0**  
- F1 Score: **1.0**

These are perfect scores. `1.0` means 100%.

Python prints:

- `1.0` → float  
- `100%` → percentage format

Both mean the same thing. Because dataset is perfectly separable, sklearn found the exact decision boundary.

---

## 🖼️ 8. Decision Boundary Plot

Save this image as:

```
plot.png
```

Interpretation:

- Red = Class 0  
- Blue = Class 1  
- Green dashed line = decision boundary (x ≈ 5.0)

Perfect separation → perfect metrics.

---

## 🏁 9. Full Summary

You now understand:

- ✔ Why logistic regression is used → For binary classification using probability.  
- ✔ What sigmoid does → Maps any value to 0–1 probability.  
- ✔ How model decides class → Threshold at probability = 0.5.  
- ✔ Why your accuracy = 1.0 → Dataset is perfectly separable.  
- ✔ Why we calculate precision, recall, F1 → Deeper insight for real-world (imperfect) datasets.  
- ✔ Why scratch project didn’t include them → Focused on core algorithm, not evaluation engineering.

---

## 🌍 10. Real-World Example

**Email Spam Classification**

| Email Feature: number of suspicious words | Prediction |
|-------------------------------------------|------------|
| value > threshold → spam | 1 |
| value < threshold → not spam | 0 |

Logistic regression:

- Learns which features increase spam probability  
- θ = weights → importance of each word  
- b = bias → base likelihood  
- sigmoid → outputs spam probability

Same process as your synthetic x>5 example.

---

## 🚀 11. Next Project (Project 7 – Softmax Regression)

Now that binary classification is complete, next is multiclass classification:

- Predict digit 0–9  
- Predict sentiment (positive / neutral / negative)  
- Predict iris flower species  

**Softmax is the foundation for neural networks.**