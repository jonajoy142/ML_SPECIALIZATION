# Project 12 — Decision Tree Classifier

A comprehensive implementation of Decision Tree Classifier using scikit-learn, designed to understand rule-based machine learning models and their interpretability.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Concepts Covered](#concepts-covered)
- [How Decision Trees Work](#how-decision-trees-work)
- [Model Configuration](#model-configuration)
- [Evaluation Metrics](#evaluation-metrics)
- [Example Results](#example-results)
- [Code Walkthrough](#code-walkthrough)
- [Comparison with Other Models](#comparison-with-other-models)
- [Real-World Applications](#real-world-applications)
- [Limitations](#limitations)
- [Why Decision Trees Matter](#why-decision-trees-matter)
- [Next Steps](#next-steps)

## 🎯 Overview

This project implements a Decision Tree Classifier using scikit-learn. Decision Trees are rule-based models that learn a sequence of if–else decisions to classify data. They are one of the most interpretable machine learning models and are heavily used in real-world ML systems.

**Key Learning Objectives:**
- Understanding what a Decision Tree is (intuition + math)
- How a tree decides where to split
- Entropy, Gini Impurity, and Information Gain
- Why trees are interpretable
- How overfitting happens in trees
- How max depth controls complexity
- How trees are used in industry
- How trees lead to Random Forests & XGBoost

## ✨ Features

- **Rule-Based Classification**: Implements decision tree using if-else logic
- **High Interpretability**: Visual tree structure showing decision paths
- **Gini Impurity**: Uses Gini impurity for split decisions
- **Overfitting Control**: Configurable max depth to prevent overfitting
- **Synthetic Data**: Controlled dataset for learning and visualization
- **Model Persistence**: Saves trained model for reuse
- **Comprehensive Evaluation**: Accuracy and detailed classification report
- **Tree Visualization**: Visual representation of decision rules

## 📦 Requirements

- Python 3.7+
- scikit-learn
- NumPy
- Matplotlib (for visualization)
- joblib (for model persistence)

## 📂 Project Structure

```
project12_decision_tree/
│
├── data.py                 # Synthetic dataset generation
├── train.py                # Train and save model
├── eval.py                 # Accuracy and classification report
├── plot.py                 # Tree visualization
├── decision_tree.png       # Saved tree image
└── README.md               # Project documentation
```

## 🚀 Usage

### Training the Model

```bash
python train.py
```

This script will:
- Generate synthetic data
- Split data into train/test sets
- Train the Decision Tree classifier
- Save the trained model
- Display tree depth and number of leaves

### Evaluating the Model

```bash
python eval.py
```

This script will:
- Load the trained model
- Make predictions on test data
- Calculate accuracy
- Display detailed classification report

### Visualizing the Tree

```bash
python plot.py
```

This script will:
- Load the trained model
- Generate and save tree visualization
- Display the decision tree structure

## 📚 Concepts Covered

### What Is a Decision Tree? (Simple Explanation)

A Decision Tree works like a flowchart.

**Example (loan approval):**
```
Is income > 50k?
 ├── Yes → Is credit score > 700?
 │        ├── Yes → Approve loan
 │        └── No  → Reject loan
 └── No  → Reject loan
```

- Each node asks a question
- Each branch is an answer
- Each leaf is a final decision

### Why Decision Trees Exist

**Other models (Logistic Regression, SVM):**
- Are mathematical
- Hard to explain to non-technical people

**Decision Trees:**
- Mimic human decision-making
- Are easy to explain
- Are transparent
- Work well on tabular data

That's why banks, hospitals, and businesses love them.

### How Andrew Ng Teaches Decision Trees

Andrew Ng introduces Decision Trees after Logistic Regression & SVM because:
- They solve different problems
- They trade accuracy for interpretability
- They form the foundation of:
  - Random Forest
  - Gradient Boosting
  - XGBoost (industry standard)

Decision Trees are a bridge between simple ML and advanced ensemble models.

## 🌳 How Decision Trees Work

### Step 1: Choose a Feature to Split

The tree tries all features and asks:
> "Which split makes the data most pure?"

### Step 2: Measure Impurity

Two common metrics:

**🔹 Gini Impurity (used here)**
- Measures how mixed the classes are
- Gini = 0 → perfectly pure
- Higher Gini → more mixed

**🔹 Entropy (alternative)**
- Based on information theory
- Both aim to separate classes cleanly

### Step 3: Information Gain

The tree chooses the split that reduces impurity the most. This is why trees are greedy algorithms.

### Core Theory

**Training Process:**
1. Load dataset
2. Split into train/test
3. Train Decision Tree
4. Tree finds best splits
5. Stops at configured depth
6. Saves the trained model

## ⚙️ Model Configuration

**Key parameters used:**

```python
DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)
```

**Why `max_depth=3`?**
- Prevents overfitting
- Forces simpler rules
- Makes tree interpretable

**Key Concept: `max_depth`**
- Limits how deep the tree can grow
- Prevents overfitting
- Depth = number of decision layers
- Andrew Ng emphasizes shallow trees for generalization

## 🧪 Synthetic Data — What & Why

**What is synthetic data?**
- Artificially generated data that mimics real patterns

**Why use it?**
- Controlled environment
- Easy visualization
- Known ground truth
- Perfect for learning

**Dataset properties:**
- 300 data points
- 4 input features
- 2 classes (binary classification)
- Slight overlap to make the problem realistic

**Data Generation Details:**

```python
make_classification(
    n_samples=300,      # 300 data points
    n_features=4,       # 4 input features
    n_informative=3,    # 3 features actually matter
    n_redundant=1,      # 1 noisy feature
    n_classes=2,        # binary classification
    random_state=42     # reproducible data
)
```

This mimics real-world data with useful + noisy features.

## 📊 Evaluation Metrics

The project evaluates the following metrics:

- **Accuracy**: Overall correctness of predictions
- **Precision**: How many predicted positives were correct
- **Recall**: How many actual positives were found
- **F1-score**: Balance of precision & recall

This is industry-standard evaluation.

## 📈 Example Results

### Training Output

```
Decision Tree trained and saved.
Tree depth: 3
Number of leaves: 6
```

**Interpretation:**
- ✔ Tree is shallow → controlled complexity
- ✔ 6 leaves → 6 decision rules
- ✔ 3 levels of decisions

### Evaluation Output

```
Accuracy: 0.9
```

**Interpretation:**
- 90% of predictions are correct
- Excellent for such a simple, interpretable model

### Classification Report

```
              precision    recall   f1-score
Class 0       0.85        0.97      0.91
Class 1       0.96        0.84      0.90
```

**Interpretation:**
- **Class 0**: Precision 0.85 → When model predicts 0, it's right 85% of time. Recall 0.97 → It finds almost all actual 0s
- **Class 1**: Precision 0.96 → Very reliable predictions. Recall 0.84 → Misses some positives
- This tradeoff is normal and expected

### Decision Tree Visualization

The tree structure is plotted and saved as `decision_tree.png`.

**This plot shows:**
- Feature splits
- Threshold values
- Class decisions
- Leaf nodes

👉 This is why Decision Trees are highly interpretable. You can literally see the logic the model learned.

## 💻 Code Walkthrough

### `data.py` — Data Generation

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

def load_data():
    X, y = make_classification(
        n_samples=300,
        n_features=4,
        n_informative=3,
        n_redundant=1,
        n_classes=2,
        random_state=42
    )
    return train_test_split(X, y, test_size=0.2, random_state=42)
```

**Key Functions:**
- `make_classification` → creates synthetic labeled data
- `train_test_split` → splits data into train/test (80/20 split)

### `train.py` — Training the Decision Tree

```python
from sklearn.tree import DecisionTreeClassifier
from data import load_data
import joblib

X_train, X_test, y_train, y_test = load_data()

model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

model.fit(X_train, y_train)
joblib.dump(model, "decision_tree.joblib")
```

**Training Step:**
- Tree learns rules
- Chooses features that reduce impurity (Gini/Entropy)
- Saves trained model to file for reuse

### `eval.py` — Evaluation

```python
from sklearn.metrics import accuracy_score, classification_report
import joblib
from data import load_data

model = joblib.load("decision_tree.joblib")
X_train, X_test, y_train, y_test = load_data()

y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

**Evaluation Process:**
- Loads trained model
- Tree makes predictions by following decision rules
- Calculates accuracy and detailed metrics

### `plot.py` — Visualizing the Tree

```python
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import joblib

model = joblib.load("decision_tree.joblib")

plt.figure(figsize=(16, 8))
plot_tree(
    model,
    filled=True,
    feature_names=["f1", "f2", "f3", "f4"],
    class_names=["Class 0", "Class 1"]
)
plt.savefig("decision_tree.png")
plt.show()
```

**What this shows:**
- Each node = a decision
- Colors = class dominance
- Gini = impurity
- Samples = data count
- Value = class distribution

## ⚖️ Comparison with Other Models

| Aspect | Logistic Regression | Decision Tree |
|--------|---------------------|---------------|
| Model type | Linear | Rule-based |
| Interpretability | Medium | **VERY HIGH** |
| Handles non-linearity | ❌ | ✅ |
| Feature scaling | Needed | Not needed |
| Overfitting risk | Low | High (if deep) |

## 🏭 Real-World Applications

Decision Trees are used in:

**💰 Banking**
- Loan approval
- Credit scoring
- Fraud detection rules

**🏥 Healthcare**
- Disease diagnosis
- Risk assessment
- Treatment decision paths

**🛒 Business**
- Customer segmentation
- Churn prediction
- Pricing rules

**🤖 ML Systems**
- Base learner in Random Forest
- Core unit in XGBoost

## ⚠️ Limitations

- Overfit easily if too deep
- Unstable (small data change → big tree change)
- Not always best accuracy

👉 That's why we move to Random Forest & XGBoost next.

## 🌟 Why Decision Trees Matter for ML Mastery

Decision Trees are the base for:
- Random Forest 🌲🌲🌲
- Gradient Boosting
- XGBoost (industry standard)
- LightGBM
- CatBoost

**Master trees → master tabular ML.**

## 🧠 Key Takeaways

You now understand:
- How decision trees work internally
- Why entropy/Gini matters
- How interpretability differs from accuracy
- Why Andrew Ng teaches trees at this stage
- How trees form the foundation of ensembles

## 🚀 Next Steps

**Project 13 — Random Forest Classifier**

The next project will focus on:
- Many trees combined
- Reduces overfitting
- Strong industry baseline
- Building ensemble intuition

---

**Note**: This project is part of a Machine Learning Specialization series designed to build foundational understanding through hands-on implementation.
