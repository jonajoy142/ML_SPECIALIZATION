Project 1 — Linear Regression From Scratch

This project implements Linear Regression without sklearn, using only NumPy.
It builds strong intuition for how ML works under the hood.

🎯 Concepts Learned
✔ Linear Regression Basics

Hypothesis function:
hθ(x) = θx + b

Meaning of theta (slope) and bias (intercept)

What a feature is and how X.shape determines data dimensions

✔ Gradient Descent

Why gradient descent is needed

Parameter update rules:

θ := θ - α * dθ
b := b - α * db


Role of learning rate

Meaning of iterations / epochs

✔ Evaluation Metrics

MSE (Mean Squared Error)

RMSE

MAE

R² Score

Inference Time

Cost Decreasing Plot (Loss Curve)

✔ Synthetic Data Generation

Why synthetic data is useful for learning

What np.random.seed(42) means (reproducibility)

How noise affects learned parameters (bias shifting)

❓ Common Questions You Asked (and Understood)
1️⃣ What is np.random.seed(42)?

Makes randomness repeatable

Ensures same synthetic data every run

Without it → different results every time

2️⃣ What is X.shape?

Shows dataset dimensions

Example: (100, 1) → 100 samples, 1 feature

Needed for correct weight initialization

3️⃣ Why do we add noise in synthetic data?

To simulate real-world imperfect data

Makes regression realistic

Explains why learned bias ≠ exactly true bias

4️⃣ Why use y_pred = dot(X, theta) + bias?

This is the linear regression equation

First predictions are usually wrong → gradient descent fixes them

5️⃣ Why these gradients?
d_theta = (1/n) * np.dot(X.T, (y_pred - y))
d_bias  = (1/n) * np.sum(y_pred - y)


Because these are partial derivatives of MSE.
They show how to adjust θ and b to reduce error.

6️⃣ Why eval.py does NOT use train.py's trained model?

eval.py re-trains a new model using the same data

That’s why results look similar

Later we will add:

model saving

model loading

inference script

7️⃣ What is inference speed?

Time taken for prediction

Linear regression is extremely fast (microseconds)

🧪 Example Evaluation Output
--- Evaluation Metrics ---
MSE: 20.50
RMSE: 4.52
MAE: 3.57
R²: 0.9895
Inference Time: 0.0000047 seconds


Interpretation:

Slope ≈ 3.03

Bias ≈ 8.42 (noise pulled it down)

R² ≈ 0.99 → excellent fit

Very fast inference time

🚀 How to Run the Project
Train Model
python train.py

Evaluate Model
python eval.py


Outputs:

Evaluation metrics

Cost (loss) curve

Learned regression line

📂 Project Folder Structure
project1_linear_regression_scratch/
│── data.py               # synthetic dataset
│── model.py              # scratch regression model
│── train.py              # trains the model
│── eval.py               # evaluates + plots
│── README.md             # documentation

🌟 Next Step: Project 2 — Linear Regression using sklearn

In the next project, we will:

Compare scratch model vs sklearn

Use real-world datasets

Measure performance differences

Validate correctness