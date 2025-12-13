# Project 9 — KNN Classification From Scratch

A comprehensive implementation of K-Nearest Neighbors (KNN) classification from scratch using only NumPy, designed to understand distance-based, non-parametric learning algorithms.

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

K-Nearest Neighbors (KNN) is a simple but powerful classification algorithm that makes predictions based on the **nearest neighbors** in the feature space, rather than learning explicit model parameters through training equations.

Unlike parametric methods (like linear/logistic regression) that learn weights and biases, KNN is a **non-parametric**, **instance-based** learning algorithm. It stores all training data and makes predictions by finding the K closest training examples.

## ✨ Features

- **Pure NumPy Implementation**: No external ML libraries required for core algorithm
- **Distance-Based Learning**: Euclidean distance computation for neighbor finding
- **Majority Voting**: Class prediction through neighbor voting mechanism
- **Configurable K Value**: Adjustable number of neighbors for prediction
- **Decision Boundary Visualization**: Plotting of classification regions
- **Synthetic Data Generation**: Using `make_blobs` for clean classification datasets

## 📦 Requirements

- Python 3.7+
- NumPy
- Matplotlib (for visualization)
- scikit-learn (for `make_blobs` dataset generation and `train_test_split` only)
- collections.Counter (built-in Python module)

## 📂 Project Structure

```
project9_knn_classification_scratch/
│
├── data.py               # Synthetic dataset generation and train/test split
├── model.py              # KNN classifier implementation
├── train.py              # Model training script
├── eval.py               # Model evaluation and accuracy calculation
├── plot.py               # Decision boundary visualization
├── knn.png               # Generated visualization
└── README.md             # Project documentation
```

## 🚀 Usage

### Training the Model

```bash
python train.py
```

This script will:
- Load and split the dataset into training and test sets
- Initialize the KNN model with specified K value
- Store training data (KNN "training" is just storing the data)
- Display model configuration

### Evaluating the Model

```bash
python eval.py
```

This script will:
- Load the trained model
- Make predictions on test data
- Calculate evaluation metrics (accuracy)
- Display performance results

### Visualizing Decision Boundaries

```bash
python plot.py
```

This script will:
- Generate decision boundary plots
- Visualize data points and classification regions
- Display the KNN decision boundaries
- Save the plot as `knn.png`

## 📚 Concepts Covered

### KNN Intuition

**Simple Analogy:**

To decide who you are, KNN looks at your **closest friends** (neighbors). If most of them are similar → you are similar too.

> "Tell me who your neighbors are, and I'll tell you who you are."

**Child-Friendly Explanation:**

Imagine:
- You are new in class
- You don't know if you are good at math
- You ask your **closest 3 friends** (K=3)
- If most are good at math → you are probably good too

That's KNN in action!

### How KNN Works

**Algorithm Steps:**

1. **Store all training data**: KNN doesn't "learn" parameters; it memorizes the training examples
2. **For a new point**:
   - Measure distance to every training point
   - Pick **K nearest** neighbors
   - Take a **majority vote** among those K neighbors
3. **Output the class**: The class that appears most frequently among K neighbors

**Key Difference from Parametric Methods:**

- **Parametric (e.g., Logistic Regression)**: Learns weights/biases through training, discards training data
- **Non-parametric (KNN)**: Stores all training data, no explicit training phase, predictions based on stored instances

### Distance Metric: Euclidean Distance

**Formula:**

For two points in n-dimensional space:

```
distance = √[(x₁ - x₂)² + (y₁ - y₂)² + ... + (z₁ - z₂)²]
```

For 2D points:

```
distance = √[(x₁ - x₂)² + (y₁ - y₂)²]
```

**Intuition:**
- Measures straight-line distance between points
- Closer points = more similar
- Points with smaller distance = stronger influence on prediction

**Implementation:**

```python
def _distance(self, x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))
```

### Choosing K Value

**What is K?**

K = number of neighbors to consider for prediction.

**Impact of K:**

- **Small K (e.g., K=1, 3):**
  - More sensitive to noise
  - Complex decision boundaries
  - Higher variance, lower bias
  - Can overfit to training data

- **Large K (e.g., K=15, 20):**
  - Smoother decision boundaries
  - Less sensitive to noise
  - Lower variance, higher bias
  - May underfit and miss local patterns

- **Common Choices:** K = 3, 5, 7 (often good starting points)

**Rule of Thumb:**
- Use odd K values to avoid ties in binary classification
- K should be much smaller than the number of training samples
- Cross-validation helps choose optimal K

**In this project:** K = 3 (default value)

### Example: How KNN Predicts

**Scenario:** K = 3

Given a new point, KNN finds the 3 nearest neighbors:

```
Neighbors → [Class_A, Class_A, Class_B]
```

**Majority Vote:**
- Class_A appears 2 times
- Class_B appears 1 time
- **Prediction → Class_A** (majority wins)

### Dataset Used

**Synthetic Data with make_blobs:**

This project uses synthetic multiclass data to clearly demonstrate:
- Distance-based learning
- Effect of K on decision boundaries
- Non-parametric classification behavior

**Example Dataset Structure:**

| Point | Features | Class |
|-------|----------|-------|
| (1, 2) | [1, 2] | 0 |
| (2, 3) | [2, 3] | 0 |
| (3, 3) | [3, 3] | 0 |
| (6, 5) | [6, 5] | 1 |
| (7, 7) | [7, 7] | 1 |
| (8, 6) | [8, 6] | 1 |

The dataset is split into training and test sets using `train_test_split`.

## 📊 Evaluation Metrics

### Accuracy

**Definition:**

```
Accuracy = (Number of correct predictions) / (Total number of predictions)
```

**Interpretation:**
- Measures the proportion of correctly classified instances
- Range: 0.0 to 1.0 (higher is better)
- Simple and intuitive metric

**When to Use:**
- Balanced datasets (equal class distribution)
- When all classes are equally important
- Quick performance assessment

**Limitations:**
- Can be misleading with imbalanced classes
- Doesn't show performance per class
- Doesn't distinguish between types of errors

For this project's clean, balanced synthetic data, accuracy is a suitable metric.

## 📊 Example Results

### Training Output

KNN training is essentially just storing the data:

```
KNN Model initialized with K = 3
Training data stored: 240 samples
```

**Note:** Unlike parametric models, KNN doesn't learn weights. "Training" means storing the training instances.

### Evaluation Output

Example evaluation results:

```
Accuracy: 0.95
```

**Interpretation:**
- Accuracy = 0.95 means 95% of test instances were correctly classified
- The model correctly predicted the class for 95 out of 100 test points

### Decision Boundaries

The visualization (`knn.png`) shows:
- Data points colored by their true class
- Decision boundaries determined by KNN
- Classification regions based on nearest neighbors

The decision boundaries are typically:
- Piecewise linear or curved
- Dependent on the K value
- More complex with smaller K, smoother with larger K

## ❓ FAQ

### 1. Why is KNN called "lazy learning"?

KNN is a **lazy learner** (also called instance-based learner) because:
- It doesn't learn a model during training
- It simply stores all training data
- All computation happens at prediction time
- Contrast with **eager learners** (like logistic regression) that learn a model during training

### 2. What are the advantages of KNN?

- **Simple and intuitive**: Easy to understand and implement
- **No training phase**: No complex optimization needed
- **Non-linear boundaries**: Can learn complex decision boundaries
- **Versatile**: Works for classification and regression
- **No assumptions**: Doesn't assume data distribution

### 3. What are the limitations of KNN?

- **Slow for large datasets**: Must compute distance to all training points for each prediction
- **Memory intensive**: Stores all training data
- **Sensitive to irrelevant features**: All features contribute equally to distance
- **Sensitive to noise**: Outliers can significantly affect predictions (especially with small K)
- **Needs feature scaling**: Features with larger scales dominate distance calculations
- **Curse of dimensionality**: Performance degrades in high-dimensional spaces

### 4. How does KNN handle ties in voting?

With even K values or when classes tie:
- The implementation typically uses the first class encountered
- Better practice: Use odd K values to avoid ties
- Some implementations use weighted voting (closer neighbors have more weight)

### 5. Should features be scaled for KNN?

**Yes!** Feature scaling is crucial for KNN because:
- Euclidean distance is affected by feature scales
- Features with larger values dominate the distance calculation
- Example: If one feature ranges 0-1 and another 0-1000, the second feature dominates
- **Solution**: Use standardization (z-score) or normalization (min-max scaling)

### 6. How do I choose the right K value?

**Methods:**
1. **Cross-validation**: Try different K values and pick the one with best validation performance
2. **Rule of thumb**: Start with K = √n (where n is number of training samples)
3. **Odd values**: Use odd K to avoid ties in binary classification
4. **Domain knowledge**: Consider what makes sense for your problem

**General guidance:**
- Small K: Good for non-linear patterns, but sensitive to noise
- Large K: Good for reducing overfitting, but may miss local patterns
- Start with K=3 or K=5 and adjust based on results

### 7. Why is KNN slow for large datasets?

For each prediction, KNN must:
1. Calculate distance to **every** training point
2. Sort all distances to find K nearest
3. This is O(n) per prediction, where n is number of training samples

**Solutions:**
- Use approximate nearest neighbor algorithms (e.g., KD-tree, Ball tree)
- Reduce training set size (but may hurt accuracy)
- Use dimensionality reduction
- For very large datasets, consider parametric methods

### 8. How does KNN compare to parametric methods?

| Aspect | KNN | Parametric (e.g., Logistic Regression) |
|--------|-----|----------------------------------------|
| Training | Stores data (O(1)) | Learns parameters (iterative) |
| Prediction | Computes distances (O(n)) | Matrix multiplication (O(d)) |
| Memory | Stores all training data | Stores only parameters |
| Interpretability | Low (black box) | Higher (can inspect weights) |
| Assumptions | None | Assumes data distribution |
| Boundaries | Can be complex | Limited by model structure |

## 🌍 Real-World Use Cases

KNN is used in various applications:

1. **Recommendation Systems**: "Users who bought X also bought Y" (collaborative filtering)
2. **Medical Diagnosis**: Classifying patients based on similar cases
3. **Image Classification**: Classifying images based on similar images (for small datasets)
4. **Pattern Recognition**: Handwriting recognition, facial recognition
5. **Fraud Detection**: Identifying fraudulent transactions based on similar patterns
6. **Text Classification**: Classifying documents based on similar documents
7. **Gene Expression Analysis**: Classifying samples based on similar gene patterns

**Note:** For very large-scale problems, KNN is often replaced by more efficient parametric methods or approximate nearest neighbor algorithms.

## 🌟 Next Steps

**Key Learnings from This Project:**

- **Distance-based learning**: Understanding how proximity determines classification
- **Non-parametric methods**: Learning algorithms that don't learn explicit parameters
- **Effect of K**: How the number of neighbors affects predictions
- **Geometry-based classification**: Classification based on spatial relationships

**Comparison with Previous Projects:**

- **Project 7 & 8 (Softmax Regression)**: Parametric, learns weights, fast prediction
- **Project 9 (KNN)**: Non-parametric, stores data, slower prediction but flexible boundaries

**Next Project Options:**

1. **KNN using sklearn**: Compare scratch implementation with scikit-learn's optimized version
2. **SVM Classifier**: Learn about support vector machines and maximum margin classification
3. **Decision Trees**: Explore tree-based classification methods
4. **Advanced Topics**: Dimensionality reduction, ensemble methods, etc.

---

**Note**: This project is part of a Machine Learning Specialization series designed to build foundational understanding through hands-on implementation. KNN demonstrates that not all learning algorithms need to learn explicit parameters—sometimes, storing and comparing instances is enough!
