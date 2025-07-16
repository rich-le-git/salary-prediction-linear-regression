# Linear Regression with Gradient Descent - Salary Prediction

This project demonstrates a simple **linear regression** model implemented from scratch using **gradient descent** to predict salaries based on years of experience. The model is trained and tested on a small dataset and visualizes the predictions compared to actual salaries.

---

## Dataset

The dataset `salary_data.csv` contains two columns:

- `YearsExperience`: Number of years of professional experience.
- `Salary`: Corresponding salary in USD.

Sample data:

| YearsExperience | Salary |
|-----------------|---------|
| 1.1             | 39343   |
| 1.3             | 46205   |
| 1.5             | 37731   |
| ...             | ...     |

---

## Description

- The dataset is loaded using `pandas`.
- Data is split into training and testing sets using `scikit-learn`.
- Gradient descent algorithm is implemented to find the best-fitting line:
  - Parameters initialized as slope (`m`) and intercept (`b`).
  - Iteratively updated based on gradients of mean squared error.
- Results are plotted using `matplotlib` to show:
  - Training data and model prediction line.
  - Testing data and model prediction line.
- Final slope and intercept values are printed.

---

## Usage

### Requirements

- Python 3.x
- numpy
- pandas
- matplotlib
- scikit-learn

Install required packages via pip:

```bash
pip install numpy pandas matplotlib scikit-learn
