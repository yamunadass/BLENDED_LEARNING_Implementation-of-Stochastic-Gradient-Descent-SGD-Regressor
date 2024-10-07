# BLENDED_LEARNING
# Implementation-of-Stochastic-Gradient-Descent-SGD-Regressor

## AIM:
To write a program to implement Stochastic Gradient Descent (SGD) Regressor for linear regression and evaluate its performance.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset
2. Preprocess the dataset
3. Train the Stochastic Gradient Descent (SGD) Regressor model
4. Visualize the result

## Program:
## Developed by: Yamuna M
## RegisterNumber: 212223230248
```
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

data = pd.read_csv('car_price_prediction_.csv')

print(data.head())

X = data[['Year', 'Engine Size', 'Mileage', 'Condition']] 
y = data['Price']

X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

sgd_model = SGDRegressor(max_iter=1000, tol=1e-3)

sgd_model.fit(X_train, y_train)

y_pred = sgd_model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse}')
print(f'R-squared: {r2}')

cv_scores = cross_val_score(sgd_model, X, y, cv=5)  # 5-fold cross-validation
print(f'Cross-Validation Scores: {cv_scores}')
print(f'Average Cross-Validation Score: {np.mean(cv_scores)}')

plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted Prices')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)  # Line for perfect prediction
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Car Prices using SGD Regressor')
plt.legend()
plt.show()
```

## Output:
![image](https://github.com/user-attachments/assets/526d4aec-210a-4380-9c59-acafee7713f0)

![image](https://github.com/user-attachments/assets/26d0fa87-2da4-4e66-a963-f3aeef620c9a)


## Result:
Thus, the implementation of Stochastic Gradient Descent (SGD) Regressor for linear regression has been successfully demonstrated and verified using Python programming.
