# Perceptron

This folder contains a from-scratch implementation of the **Perceptron**,  
the simplest neural network and one of the earliest algorithms for binary classification.

---

## Model Equation

The perceptron computes:

$$
z = Xw + b
$$

Then applies the **step activation**:

$$
\hat{y} =
\begin{cases}
1 & \text{if } z \ge 0 \\
0 & \text{otherwise}
\end{cases}
$$

---

## Perceptron Learning Rule

Weights and bias are updated only when the model makes a mistake:

$$
w = w + \eta (y - \hat{y}) x
$$

$$
b = b + \eta (y - \hat{y})
$$

Where  
- \( \eta \) = learning rate  
- \( y \) = true label  
- \( \hat{y} \) = predicted label  

---

## Files

- `perceptron.py`
- `README.md`

