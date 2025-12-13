📘 Project 13 — Random Forest Classifier

Random Forest is one of the most powerful and widely used classical machine learning algorithms for tabular data.

It improves upon Decision Trees by using many trees together instead of relying on just one.

🌲 What is Random Forest? (Simple Explanation)

A Decision Tree is like one person making a decision

A Random Forest is like asking 100 people and taking a vote

👉 One tree can be wrong
👉 Many trees together are much more reliable

Random Forest = Many Decision Trees + Majority Voting

❓ Why Random Forest is Needed
Problem with Decision Trees

Decision Trees:

❌ Overfit easily

❌ Memorize training data

❌ Very sensitive to small data changes

How Random Forest Fixes This

Random Forest:

✔ Trains many trees

✔ Each tree sees different data (bootstrapping)

✔ Each tree sees different features

✔ Final prediction = majority vote

This leads to:

✅ Higher accuracy

✅ Better generalization

✅ Much less overfitting

🧠 How Random Forest Works (High Level)

Randomly sample data (with replacement)

Train a decision tree on each sample

Limit tree depth to avoid memorization

Each tree makes a prediction

Final prediction = majority vote

This is called Bagging (Bootstrap Aggregation).

🧪 Dataset Used (Synthetic Data)

We use synthetic data generated with make_classification.

Why synthetic?

Controlled environment

Known class boundaries

Perfect for learning & visualization

Dataset properties:

300 data points

2 features (easy to plot)

2 classes (binary classification)

Train/Test split = 80/20

This simulates real-world structured data like:

customer profiles

credit risk data

medical measurements

📂 Project Structure
project13_random_forest/
│── data.py          # generate synthetic dataset
│── train.py         # train & save model
│── eval.py          # evaluate performance
│── plot.py          # decision boundary visualization
│── RF.png           # saved plot
│── README.md

⚙️ Model Configuration

We train the model using:

RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42
)

Meaning:

n_estimators=100 → 100 decision trees

max_depth=5 → prevents overfitting

random_state=42 → reproducible results

📊 Results (Your Output)
Training
Random Forest trained and saved.

Evaluation
Accuracy: 0.95


Classification Report:

Class	Precision	Recall	F1-score
0	0.94	0.97	0.95
1	0.97	0.93	0.95

✔ High accuracy
✔ Balanced precision & recall
✔ Much better than a single decision tree

📈 Visualization

The decision boundary is saved as:

RF.png


Add this to README:

![Random Forest Decision Boundary](./RF.png)


What the plot shows:

Smooth boundaries

Less noise than Decision Tree

Strong generalization

🔍 Random Forest vs Decision Tree vs Logistic Regression
Feature	Logistic Regression	Decision Tree	Random Forest
Model Type	Linear	Rule-based	Ensemble
Handles Non-linearity	❌ No	✅ Yes	✅✅ Yes
Overfitting	Low	High	Very Low
Accuracy	Medium	Medium	High
Interpretability	Medium	High	Medium
Industry Usage	High	Medium	Very High
🌍 Real-World Uses of Random Forest

Random Forest is heavily used in industry for:

🏦 Credit risk scoring

💳 Fraud detection

🏥 Medical diagnosis

📉 Customer churn prediction

📊 Tabular business data

🏆 Kaggle competitions (baseline model)

Andrew Ng’s advice:
“If you don’t know what model to try first on tabular data — use Random Forest.”

🧠 Why Andrew Ng Teaches This After Decision Trees

Andrew Ng’s teaching order:

Linear models

Logistic regression

Decision trees

Random Forest

Boosting (XGBoost)

Neural Networks

Because Random Forest:

Builds ensemble intuition

Fixes tree weaknesses

Bridges classical ML → advanced ML

Is extremely practical

✅ What You Mastered in Project 13

✔ Ensemble learning
✔ Bagging (bootstrap aggregation)
✔ Reducing overfitting
✔ Stability vs variance
✔ Industry-grade ML modeling

🔜 What’s Next?
▶ Project 14 — XGBoost / Gradient Boosting

More powerful than Random Forest

Sequential learning

Industry & Kaggle standard

Strong regularization

After that → Neural Networks 🚀

If you want, next I can:

✅ Start Project 14 — XGBoost