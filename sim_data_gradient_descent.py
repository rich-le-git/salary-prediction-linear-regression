import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import pandas as pd  
salary_data = pd.read_csv('salary_data.csv')

X, y = salary_data[['YearsExperience']].values, salary_data[['Salary']].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Gradient Descent for Linear Regression
learning_rate = 0.02
n_iterations = 1000
m, b = 0, 0  # initial coefficients

# Gradient Descent Loop
for iteration in range(n_iterations):
    # Predictions
    y_pred = m * X_train + b
    # Errors
    error = y_pred - y_train
    # Gradients
    m_gradient = (2 / len(X_train)) * np.sum(X_train * error)
    b_gradient = (2 / len(X_train)) * np.sum(error)
    # Update coefficients
    m -= learning_rate * m_gradient
    b -= learning_rate * b_gradient

# Predictions
y_test_pred = m * X_test + b
y_train_pred = m * X_train + b

# Plotting the results
plt.figure(figsize=(12, 6))

# Training data plot
plt.subplot(1, 2, 1)
plt.scatter(X_train, y_train, color="red", label="Training Observations")
plt.plot(X_train, y_train_pred, color="green", label="Ridge Model Predictions")
plt.title("Training Data with Model Predictions")
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.legend()

# Testing data plot
plt.subplot(1, 2, 2)
plt.scatter(X_test, y_test, color="red", label="Testing Observations")
plt.plot(X_test, y_test_pred, color="green", label="Ridge Model Predictions")
plt.title("Testing Data with Model Predictions")
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.legend()

plt.tight_layout()
plt.show()

# Output the coefficients
print("Slope (m):", m)
print("Intercept (b):", b)
