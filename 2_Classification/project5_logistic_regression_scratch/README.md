# 📘 Project 5 — Logistic Regression (From Scratch)

This project implements **Logistic Regression manually using NumPy**, without sklearn. It teaches how binary classification works **under the hood** at the most fundamental level.

---

## 📋 Table of Contents

- [Overview](#overview)
- [What You Will Learn](#what-you-will-learn)
- [Why Logistic Regression Exists](#why-logistic-regression-exists)
- [Sigmoid Function](#sigmoid-function)
- [Model Formula](#model-formula)
- [Q&A (Your Questions Answered)](#qa-your-questions-answered)
- [Example of Prediction](#example-of-prediction)
- [Decision Boundary Plot](#decision-boundary-plot)
- [Your Actual Results](#your-actual-results)
- [Real World Example — Spam Classification](#real-world-example--spam-classification)
- [Folder Structure](#folder-structure)
- [How to Run](#how-to-run)
- [Next Project](#next-project)

---

## Overview

Goal: understand and implement Logistic Regression from scratch using NumPy for binary classification, including the math, training loop, and evaluation.

---

## 🎯 What You Will Learn

### ✔ 1. The difference between **Linear Regression** and **Logistic Regression**

| Linear Regression       | Logistic Regression        |
| ----------------------- | -------------------------- |
| Predicts numbers        | Predicts classes (0/1)     |
| Output: any real number | Output: probability 0–1    |
| Uses straight line      | Uses S-curve (sigmoid)     |
| Loss: MSE               | Loss: Binary Cross Entropy |

---

### ✔ 2. Why Logistic Regression Exists

Because Linear Regression cannot restrict outputs to **0 or 1**.

Classification requires probability → **Logistic regression uses sigmoid**. Sigmoid makes the output look like:

* near **0** → class 0  
* near **1** → class 1

---

## 🔁 Sigmoid Function

The sigmoid function:

[
\sigma(z) = \frac{1}{1 + e^{-z}}
]

It takes any number (−∞ to +∞) and compresses it to **0–1**.

**Intuition:**

* Large negative number → sigmoid ≈ 0  
* Large positive number → sigmoid ≈ 1  
* Zero → sigmoid = 0.5

This is why logistic regression can classify.

---

## 🧮 Model Formula

Logistic Regression first computes a **linear part** just like linear regression:

[
z = \theta x + b
]

Then applies sigmoid:

[
\hat{y} = \sigma(z)
]

Finally converts probability → class:

```
if probability >= 0.5 → predict class 1
else → predict class 0
```

---

## 🧠 Q&A (Your Questions Answered)

### Q1: Isn’t logistic regression same as linear regression?

✔ Same **linear core** (`θx + b`)  
❌ Different **activation**

Linear → outputs number  
Logistic → number → sigmoid → probability

---

### Q2: What does this mean?

`(X[:, 0] > 5).astype(int)`

This creates **labels y** for synthetic data:

* If feature x > 5 → label = 1  
* Else → label = 0

Like real world:

* Transaction amount > ₹5000 → fraud (1)  
* Else → not fraud (0)

---

### Q3: Explain this part of training:

```python
for _ in range(self.iterations):
    linear = np.dot(X, self.theta) + self.bias
    y_pred = sigmoid(linear)

    d_theta = (1/m) * np.dot(X.T, (y_pred - y))
    d_bias = (1/m) * np.sum(y_pred - y)

    self.theta -= self.lr * d_theta
    self.bias -= self.lr * d_bias
```

🔹 Compute z = θx + b  
🔹 Convert z → probability using sigmoid  
🔹 Compute gradient of loss  
🔹 Update parameters  
🔹 Repeat (gradient descent)

Exactly like Linear Regression, except:

👉 Instead of raw prediction, we apply **sigmoid**.

---

### Q4: Why accuracy only? Should we check more metrics?

For perfectly separable synthetic data:

* No noise  
* Clean boundary at x = 5

Accuracy = enough.

Later real datasets need:

* Precision  
* Recall  
* F1  
* ROC-AUC

---

### Q5: Your understanding?:

> logistic regression is like linear regression but outputs 0/1 via sigmoid and threshold

✅ YES — this is correct.

Final refinement:

* Logistic regression predicts **probability**, not just a number.  
* Decision boundary happens where probability = 0.5.

---

## 🔢 Example of Prediction (Simple Numbers)

Take:

θ = 1.49  
b = –7.26

Predict for x = 3:

```
z = 1.49*3 - 7.26 = -2.79
sigmoid(-2.79) = 0.057 → predicts class 0
```

Predict for x = 9:

```
z = 1.49*9 - 7.26 = 6.14
sigmoid(6.14) = 0.997 → predicts class 1
```

✔ This is why your model works.

---

## 📊 Decision Boundary Plot

Add this file to your folder as **plot5.png**:

```
2_Classification/project5_logistic_regression_scratch/plot5.png
```

Embed in README:

![Decision Boundary](./plot5.png)

Interpretation:

* Red points = class 0  
* Blue points = class 1  
* Green dashed line → decision boundary  
* Everything left of line → predicted 0  
* Everything right → predicted 1

---

## 🧪 Your Actual Results (From eval.py)

```
Theta: [1.49655377]
Bias: -7.269057951870945
Accuracy: 0.99
```

Perfect results because:

* Data is clean  
* Classification line at x ≃ 4.59  
* Sigmoid outputs sharp probability changes

---

## 🚀 Real World Example — Spam Classification

Feature: number of suspicious words  
Label: spam (1) or not spam (0)

[
z = \theta x + b
]

If z very positive → sigmoid ≈ 1 → spam  
If z very negative → sigmoid ≈ 0 → not spam

This is exactly how Gmail spam filters started originally.

---

## 📦 Folder Structure

```
project5_logistic_regression_scratch/
│── data.py
│── model.py
│── train.py
│── eval.py
│── plots.py
│── plot5.png
│── README.md
```

---

## 🚀 How to Run

### Train:

```
python train.py
```

### Evaluate:

```
python eval.py
```

### Plot decision boundary:

```
python plots.py
```

---

## 🟢 Next Project

### **Project 6 — Logistic Regression with sklearn**

We will learn:

* `sklearn.linear_model.LogisticRegression`  
* Predict probabilities  
* Plot ROC Curve  
* Multiclass = Softmax Regression

