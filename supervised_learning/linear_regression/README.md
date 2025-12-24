# Linear Regression (From Scratch)

This folder contains a clean, from-scratch implementation of **Linear Regression**  
using **Gradient Descent** as the optimization algorithm.  
No machine learning libraries (like scikit-learn) are used.

Linear Regression is one of the most fundamental supervised learning algorithms  
and serves as the foundation for more advanced ML models.

---

## What is Linear Regression?

Linear Regression models the relationship between:

- **input features (X)**  
- **target values (y)**  

by fitting a straight line:

\[
\hat{y} = Xw + b
\]

Where:
- `w` = weights
- `b` = bias

The goal is to minimize the **Mean Squared Error (MSE)** between  
predictions and true values.

---

## How It Works

### 1. Compute prediction  
\[
\hat{y} = Xw + b
\]

### 2. Compute MSE loss  
\[
L = \frac{1}{n} \sum (y - \hat{y})^2
\]

### 3. Compute gradients  
\[
\frac{\partial L}{\partial w} = -\frac{2}{n} X^T (y - \hat{y})
\]

\[
\frac{\partial L}{\partial b} = -\frac{2}{n} \sum (y - \hat{y})
\]

### 4. Update parameters using Gradient Descent  
\[
w = w - \alpha \frac{\partial L}{\partial w}
\]

\[
b = b - \alpha \frac{\partial L}{\partial b}
\]

Where `α` is the learning rate.

---

## Files

- **`linear_regression.py`**  
  Contains the LinearRegression class implemented from scratch  
  using your `GradientDescent` optimizer.

- **`README.md`**  
  Documentation for this module.

---

## Learning Outcomes

By studying this implementation you will understand:

- how gradient descent trains a model  
- how loss decreases over time  
- how linear models work internally  
- how ML frameworks structure `fit()` and `predict()`  

This prepares you for logistic regression, SVMs, and neural networks.
