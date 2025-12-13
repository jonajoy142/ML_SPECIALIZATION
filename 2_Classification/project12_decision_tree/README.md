📘 Project 12 — Decision Tree Classifier

This project implements a Decision Tree Classifier using scikit-learn.

Decision Trees are rule-based models that learn a sequence of if–else decisions to classify data.
They are one of the most interpretable machine learning models and are heavily used in real-world ML systems.

🎯 What You Learn in This Project

What a Decision Tree is (intuition + math)

How a tree decides where to split

Entropy, Gini Impurity, Information Gain

Why trees are interpretable

How overfitting happens in trees

How max depth controls complexity

How trees are used in industry

How trees lead to Random Forests & XGBoost

🌳 What Is a Decision Tree? (Simple Explanation)

A Decision Tree works like a flowchart.

Example (loan approval):

Is income > 50k?
 ├── Yes → Is credit score > 700?
 │        ├── Yes → Approve loan
 │        └── No  → Reject loan
 └── No  → Reject loan


Each node asks a question,
each branch is an answer,
each leaf is a final decision.

🧠 Why Decision Trees Exist

Other models (Logistic Regression, SVM):

Are mathematical

Hard to explain to non-technical people

Decision Trees:

Mimic human decision-making

Are easy to explain

Are transparent

Work well on tabular data

That’s why banks, hospitals, and businesses love them.

📚 How Andrew Ng Teaches Decision Trees

Andrew Ng introduces Decision Trees after Logistic Regression & SVM because:

They solve different problems

They trade accuracy for interpretability

They form the foundation of:

Random Forest

Gradient Boosting

XGBoost (industry standard)

Decision Trees are a bridge between simple ML and advanced ensemble models.

⚙️ How a Decision Tree Learns (Core Theory)
Step 1: Choose a feature to split

The tree tries all features and asks:

“Which split makes the data most pure?”

Step 2: Measure impurity

Two common metrics:

🔹 Gini Impurity (used here)

Measures how mixed the classes are.

Gini = 0 → perfectly pure

Higher Gini → more mixed

🔹 Entropy (alternative)

Based on information theory.

Both aim to separate classes cleanly.

Step 3: Information Gain

The tree chooses the split that reduces impurity the most.

This is why trees are greedy algorithms.

🧪 Synthetic Data — What & Why

This project uses synthetic data.

What is synthetic data?

Artificially generated data that mimics real patterns.

Why use it?

Controlled environment

Easy visualization

Known ground truth

Perfect for learning

The dataset contains:

2 input features

2 classes (binary classification)

Slight overlap to make the problem realistic

🧩 Model Configuration

Key parameters used:

max_depth = 3

Why?

Prevents overfitting

Forces simpler rules

Makes tree interpretable

🧠 What Happens in Training (Conceptual)

Load dataset

Split into train/test

Train Decision Tree

Tree finds best splits

Stops at depth = 3

Saves the trained model

📊 Results (Your Actual Output Explained)
Training Output
Decision Tree trained and saved.
Tree depth: 3
Number of leaves: 6


✔ Tree is shallow → controlled complexity
✔ 6 leaves → 6 decision rules

Evaluation Output
Accuracy: 0.9


This means:

90% of predictions are correct

Excellent for such a simple, interpretable model

Classification Report Explained
precision    recall   f1-score

Class 0

Precision 0.85 → When model predicts 0, it’s right 85% of time

Recall 0.97 → It finds almost all actual 0s

Class 1

Precision 0.96 → Very reliable predictions

Recall 0.84 → Misses some positives

This tradeoff is normal and expected.

📈 Decision Tree Visualization

The tree structure is plotted and saved as:

decision_tree.png


This plot shows:

Feature splits

Threshold values

Class decisions

Leaf nodes

👉 This is why Decision Trees are highly interpretable.

You can literally see the logic the model learned.

🏭 Real-World Use Cases

Decision Trees are used in:

💰 Banking

Loan approval

Credit scoring

Fraud detection rules

🏥 Healthcare

Disease diagnosis

Risk assessment

Treatment decision paths

🛒 Business

Customer segmentation

Churn prediction

Pricing rules

🤖 ML Systems

Base learner in Random Forest

Core unit in XGBoost

⚠️ Limitations of Decision Trees

Overfit easily if too deep

Unstable (small data change → big tree change)

Not always best accuracy

👉 That’s why we move to Random Forest & XGBoost next.

📂 Project Structure
project12_decision_tree/
│── data.py                 # synthetic dataset generation
│── train.py                # train & save model
│── eval.py                 # accuracy + classification report
│── plot.py                 # tree visualization
│── decision_tree.png       # saved tree image
│── README.md


data.py — Data Generation (Line-by-Line)
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


make_classification → creates synthetic labeled data

train_test_split → splits data into train/test

def load_data():


Defines a function so other files can reuse data.

X, y = make_classification(
    n_samples=300,
    n_features=4,
    n_informative=3,
    n_redundant=1,
    n_classes=2,
    random_state=42
)

What this means (VERY IMPORTANT):

n_samples=300 → 300 data points

n_features=4 → 4 input features

n_informative=3 → 3 features actually matter

n_redundant=1 → 1 noisy feature

n_classes=2 → binary classification

random_state=42 → reproducible data

👉 This mimics real-world data with useful + noisy features.

return train_test_split(X, y, test_size=0.2, random_state=42)


80% training data

20% testing data

3️⃣ train.py — Training the Decision Tree
from sklearn.tree import DecisionTreeClassifier
from data import load_data
import joblib


DecisionTreeClassifier → ML model

joblib → save trained model to disk

X_train, X_test, y_train, y_test = load_data()


Loads data from data.py

Splits into train & test

model = DecisionTreeClassifier(
    max_depth=3,
    random_state=42
)

Key concept: max_depth

Limits how deep the tree can grow

Prevents overfitting

Depth = number of decision layers

Andrew Ng emphasizes shallow trees for generalization.

model.fit(X_train, y_train)


Training step:

Tree learns rules

Chooses features that reduce impurity (Gini/Entropy)

joblib.dump(model, "decision_tree.joblib")


Saves trained model to file for reuse.

print("Decision Tree trained and saved.")
print("Tree depth:", model.get_depth())
print("Number of leaves:", model.get_n_leaves())


Your output:

Tree depth: 3
Number of leaves: 6


Meaning:

3 levels of decisions

6 final decision paths

4️⃣ eval.py — Evaluation (Line-by-Line)
from sklearn.metrics import accuracy_score, classification_report
import joblib
from data import load_data


Accuracy → overall correctness

Classification report → detailed metrics

model = joblib.load("decision_tree.joblib")


Loads trained model.

X_train, X_test, y_train, y_test = load_data()


Loads same data split.

y_pred = model.predict(X_test)


Tree makes predictions by following decision rules.

print("Accuracy:", accuracy_score(y_test, y_pred))


Your result:

Accuracy: 0.9


→ 90% correct predictions.

print(classification_report(y_test, y_pred))

Interpretation of your output:

Precision: how many predicted positives were correct

Recall: how many actual positives were found

F1-score: balance of precision & recall

This is industry-standard evaluation.

5️⃣ plot.py — Visualizing the Tree
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import joblib


plot_tree → draws decision rules visually

model = joblib.load("decision_tree.joblib")


Load trained tree.

plt.figure(figsize=(16, 8))
plot_tree(
    model,
    filled=True,
    feature_names=["f1", "f2", "f3", "f4"],
    class_names=["Class 0", "Class 1"]
)

What this shows:

Each node = a decision

Colors = class dominance

Gini = impurity

Samples = data count

Value = class distribution

plt.savefig("decision_tree.png")
plt.show()


Saves plot inside project folder.

6️⃣ Understanding Your Results
Training Output
Tree depth: 3
Leaves: 6


✔ Shallow → good generalization
✔ Not memorizing noise

Evaluation Output
Accuracy: 0.90


✔ Strong result
✔ Balanced precision & recall

7️⃣ Decision Tree vs Logistic Regression
Aspect	Logistic Regression	Decision Tree
Model type	Linear	Rule-based
Interpretability	Medium	VERY HIGH
Handles non-linearity	❌	✅
Feature scaling	Needed	Not needed
Overfitting risk	Low	High (if deep)
8️⃣ Why Decision Trees Matter for ML Mastery

Decision Trees are the base for:

Random Forest 🌲🌲🌲

Gradient Boosting

XGBoost (industry standard)

LightGBM

CatBoost

Master trees → master tabular ML.

🧠 Key Takeaways

You now understand:

How decision trees work internally

Why entropy/Gini matters

How interpretability differs from accuracy

Why Andrew Ng teaches trees at this stage

How trees form the foundation of ensembles

🚀 What’s Next (Natural Progression)
▶ Project 13 — Random Forest

Many trees combined

Reduces overfitting

Strong industry baseline