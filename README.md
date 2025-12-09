# Machine Learning Specialization — Complete Project Repository

A comprehensive mono-repository containing hands-on machine learning projects covering the entire Machine Learning Specialization curriculum (Andrew Ng + DeepLearning.AI), plus extended practical implementations.

## 📋 Table of Contents

- [Overview](#overview)
- [Repository Structure](#repository-structure)
- [Learning Path](#learning-path)
- [Projects by Module](#projects-by-module)
- [Skills Mastered](#skills-mastered)
- [Getting Started](#getting-started)
- [Resources](#resources)
- [Contributing](#contributing)

## 🎯 Overview

This repository contains **22+ portfolio-grade ML projects** organized into **7 major modules**, progressing from fundamental regression and classification to advanced topics like neural networks, unsupervised learning, recommender systems, and reinforcement learning.

### Key Features

- ✅ **Complete Coverage**: 100% of Machine Learning Specialization curriculum
- ✅ **Hands-On Learning**: Every concept implemented through practical projects
- ✅ **Multiple Approaches**: From-scratch implementations, sklearn, and TensorFlow/Keras
- ✅ **Professional Structure**: Organized like a real ML engineer portfolio
- ✅ **Comprehensive Documentation**: Each project includes detailed README with concepts, usage, and examples

## 📂 Repository Structure

```
ML_SPECIALIZATION_MONO_REPO/
│
├── 0_Notes/                          # Theory notes and external resources
│   └── notion_link.md                # Link to comprehensive Notion notes
│
├── 1_Regression/                     # Regression projects (1-4)
│   ├── project1_linear_regression_scratch/
│   ├── project2_linear_regression_sklearn/
│   ├── project3_polynomial_regression/
│   └── project4_ridge_lasso/         # (Upcoming)
│
├── 2_Classification/                 # Classification projects (5-8)
│   └── (Projects to be added)
│
├── 3_Neural_Networks/                # Neural network projects (9-12)
│   └── (Projects to be added)
│
├── 4_Advanced_ML/                    # Advanced ML projects (13-15)
│   └── (Projects to be added)
│
├── 5_Unsupervised/                   # Unsupervised learning (16-18)
│   └── (Projects to be added)
│
├── 6_Recommenders/                   # Recommender systems (19-20)
│   └── (Projects to be added)
│
└── 7_Reinforcement_Learning/        # RL projects (21-22)
    └── (Projects to be added)
```

## 🗺️ Learning Path

The projects are designed to be completed sequentially, building upon previous concepts:

1. **Regression** → Foundation in linear models and optimization
2. **Classification** → Extending to discrete outputs and decision boundaries
3. **Neural Networks** → Deep learning fundamentals
4. **Advanced ML** → Ensemble methods and tree-based models
5. **Unsupervised Learning** → Working without labels
6. **Recommender Systems** → Real-world application systems
7. **Reinforcement Learning** → Agent-based learning

## 📚 Projects by Module

### ✅ 1. Regression (Projects 1–4)

**Location**: `1_Regression/`

**Focus**: Building foundations in linear models, optimization, and model selection.

| Project | Topic | Status | Key Skills |
|---------|-------|--------|------------|
| [Project 1](1_Regression/project1_linear_regression_scratch/) | Linear Regression From Scratch | ✅ Complete | Gradient descent, MSE, parameter updates, evaluation metrics |
| [Project 2](1_Regression/project2_linear_regression_sklearn/) | Linear Regression (sklearn) | ✅ Complete | Analytical solution, API usage, performance comparison |
| [Project 3](1_Regression/project3_polynomial_regression/) | Polynomial Regression | ✅ Complete | Feature expansion, underfitting/overfitting, curve visualization |
| Project 4 | Ridge & Lasso Regression | 🚧 Upcoming | Regularization, L1/L2 penalties, hyperparameter tuning |

**Concepts Covered**:
- Linear regression fundamentals
- Gradient descent optimization
- Analytical vs. iterative solutions
- Polynomial feature expansion
- Bias-variance tradeoff
- Model evaluation metrics (MSE, RMSE, MAE, R²)
- Regularization techniques

---

### 📋 2. Classification (Projects 5–8)

**Location**: `2_Classification/`

**Focus**: Learning classification fundamentals and decision boundaries.

| Project | Topic | Status | Key Skills |
|---------|-------|--------|------------|
| Project 5 | Logistic Regression (Binary) | 🚧 Upcoming | Sigmoid function, BCE loss, probability outputs |
| Project 6 | Softmax Regression | 🚧 Upcoming | Multiclass classification, one-vs-all |
| Project 7 | KNN Classification | 🚧 Upcoming | Distance-based learning, k-nearest neighbors |
| Project 8 | SVM Classifier | 🚧 Upcoming | Margins, kernels, hyperparameter optimization |

**Concepts Covered**:
- Binary and multiclass classification
- Decision boundaries
- Evaluation metrics (precision, recall, F1-score, confusion matrix)
- Distance-based algorithms
- Support vector machines

---

### 🧠 3. Neural Networks (Projects 9–12)

**Location**: `3_Neural_Networks/`

**Focus**: Deep learning fundamentals and neural network architecture.

| Project | Topic | Status | Key Skills |
|---------|-------|--------|------------|
| Project 9 | Neural Network From Scratch | 🚧 Upcoming | Forward propagation, backpropagation, manual gradients |
| Project 10 | Neural Network (TensorFlow/Keras) | 🚧 Upcoming | Model compilation, training, evaluation |
| Project 11 | MNIST Digit Classifier | 🚧 Upcoming | Image classification, softmax output layer |
| Project 12 | Understanding Activations | 🚧 Upcoming | ReLU, Sigmoid, Tanh comparisons |

**Concepts Covered**:
- Forward and backward propagation
- Activation functions
- Loss functions
- Optimization algorithms
- Deep learning frameworks

---

### 🚀 4. Advanced ML (Projects 13–15)

**Location**: `4_Advanced_ML/`

**Focus**: Ensemble methods and advanced machine learning techniques.

| Project | Topic | Status | Key Skills |
|---------|-------|--------|------------|
| Project 13 | Decision Tree Classifier | 🚧 Upcoming | Tree construction, splitting criteria, pruning |
| Project 14 | Random Forest Classifier | 🚧 Upcoming | Ensemble learning, bagging, feature importance |
| Project 15 | XGBoost for Tabular ML | 🚧 Upcoming | Gradient boosting, hyperparameter tuning |

**Concepts Covered**:
- Tree-based algorithms
- Ensemble methods
- Bagging and boosting
- Feature importance
- Model interpretability

---

### 🔍 5. Unsupervised Learning (Projects 16–18)

**Location**: `5_Unsupervised/`

**Focus**: Learning from unlabeled data.

| Project | Topic | Status | Key Skills |
|---------|-------|--------|------------|
| Project 16 | K-Means Clustering | 🚧 Upcoming | Clustering algorithms, centroids, elbow method |
| Project 17 | PCA Dimensionality Reduction | 🚧 Upcoming | Principal components, variance explained, feature reduction |
| Project 18 | Anomaly Detection | 🚧 Upcoming | Gaussian models, outlier detection, threshold selection |

**Concepts Covered**:
- Clustering algorithms
- Dimensionality reduction
- Anomaly detection
- Unsupervised evaluation metrics

---

### 🎬 6. Recommender Systems (Projects 19–20)

**Location**: `6_Recommenders/`

**Focus**: Building recommendation systems used by major platforms.

| Project | Topic | Status | Key Skills |
|---------|-------|--------|------------|
| Project 19 | Collaborative Filtering | 🚧 Upcoming | User-item matrix, similarity metrics, matrix factorization |
| Project 20 | Content-Based Recommendation | 🚧 Upcoming | Feature extraction, similarity matching, hybrid approaches |

**Concepts Covered**:
- Collaborative filtering
- Content-based filtering
- Matrix factorization
- Hybrid recommendation systems

---

### 🎮 7. Reinforcement Learning (Projects 21–22)

**Location**: `7_Reinforcement_Learning/`

**Focus**: Agent-based learning and decision making.

| Project | Topic | Status | Key Skills |
|---------|-------|--------|------------|
| Project 21 | Basic Q-Learning | 🚧 Upcoming | Q-values, epsilon-greedy, reward functions |
| Project 22 | GridWorld Agent | 🚧 Upcoming | Environment interaction, policy learning, value iteration |

**Concepts Covered**:
- Agent-environment interaction
- Rewards and policies
- Q-learning algorithm
- Value iteration
- Exploration vs. exploitation

---

## 🎓 Skills Mastered

By completing this repository, you will master:

### Machine Learning Fundamentals
- ✅ Regression (linear, polynomial, regularized)
- ✅ Classification (binary, multiclass)
- ✅ Overfitting & regularization
- ✅ Gradient descent and optimization
- ✅ Model evaluation and selection
- ✅ Bias-variance tradeoff

### Deep Learning
- ✅ Neural network architecture
- ✅ Forward and backward propagation
- ✅ Activation functions
- ✅ Loss functions and optimization
- ✅ Deep learning frameworks (TensorFlow/Keras)

### Practical ML Engineering
- ✅ Evaluation metrics (MSE, RMSE, MAE, R², precision, recall, F1)
- ✅ Train/test/validation splits
- ✅ Hyperparameter tuning
- ✅ Feature engineering
- ✅ Model selection and comparison

### Unsupervised Learning
- ✅ Clustering algorithms (K-Means)
- ✅ Dimensionality reduction (PCA)
- ✅ Anomaly detection

### Systems-Level ML
- ✅ Recommender systems
- ✅ Reinforcement learning
- ✅ Real-world ML applications

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- NumPy
- scikit-learn
- Matplotlib
- TensorFlow/Keras (for neural network projects)
- Jupyter Notebook (optional, for experimentation)

### Installation

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd ML_SPECIALIZATION_MONO_REPO
   ```

2. **Install dependencies**:
   ```bash
   pip install numpy scikit-learn matplotlib tensorflow
   ```

3. **Navigate to a project**:
   ```bash
   cd 1_Regression/project1_linear_regression_scratch
   ```

4. **Run the project**:
   ```bash
   python train.py
   python eval.py
   ```

### Recommended Learning Path

1. Start with **Project 1** (Linear Regression From Scratch) to build intuition
2. Complete projects sequentially within each module
3. Read each project's README for detailed explanations
4. Experiment with hyperparameters and modifications
5. Compare your results with expected outputs

## 📖 Resources

### External Notes

**🔗 [Comprehensive Notion Notes](https://www.notion.so/Machine-Learning-Specialization-Master-Revision-Notes-2c07230caa57800da368fabd0c3c059d)**

Complete theory notes, visual diagrams, and explanations covering all ML Specialization topics.

### Additional Resources

- **Machine Learning Specialization** (Coursera) - Andrew Ng, DeepLearning.AI
- **scikit-learn Documentation** - [https://scikit-learn.org/](https://scikit-learn.org/)
- **TensorFlow Documentation** - [https://www.tensorflow.org/](https://www.tensorflow.org/)
- **NumPy Documentation** - [https://numpy.org/doc/](https://numpy.org/doc/)

## 📊 Project Status Overview

| Module | Projects | Completed | In Progress | Upcoming |
|--------|----------|-----------|-------------|----------|
| Regression | 4 | 3 | 0 | 1 |
| Classification | 4 | 0 | 0 | 4 |
| Neural Networks | 4 | 0 | 0 | 4 |
| Advanced ML | 3 | 0 | 0 | 3 |
| Unsupervised | 3 | 0 | 0 | 3 |
| Recommenders | 2 | 0 | 0 | 2 |
| Reinforcement Learning | 2 | 0 | 0 | 2 |
| **Total** | **22** | **3** | **0** | **19** |

## 🎯 Repository Goals

This repository serves as:

- 📚 **Learning Resource**: Comprehensive hands-on implementation of ML concepts
- 💼 **Portfolio**: Showcase of ML engineering skills
- 🔄 **Reference**: Quick access to implementations and explanations
- 🎓 **Study Guide**: Structured path through ML Specialization

## 🤝 Contributing

This is a personal learning repository, but suggestions and improvements are welcome:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This repository is for educational purposes. Feel free to use it for learning and reference.

---

**Note**: This repository is part of a comprehensive Machine Learning Specialization learning path. Each project builds upon previous concepts, creating a structured progression from fundamentals to advanced topics.

**Last Updated**: Projects are continuously being added and improved. Check individual project READMEs for the most current information.
