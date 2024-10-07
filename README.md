# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

STEP:1

Import the standard libraries.

STEP:2

Upload the dataset and check for any null values using .isnull() function.

STEP:3

Import LabelEncoder and encode the dataset.

STEP:4

Import DecisionTreeRegressor from sklearn and apply the model on the dataset.

STEP:5

Predict the values of arrays.

STEP:6

.Import metrics from sklearn and calculate the MSE and R2 of the model on the dataset.

STEP:7

Predict the values of array.

STEP:8

Apply to new unknown values.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: DILLIARASU M
RegisterNumber: 212223230049 
*/
```
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier, plot_tree

data = pd.read_csv("Salary_EX7.csv")

data.head()

data.info()

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

data["Position"] = le.fit_transform(data["Position"])

data.head()

x=data[["Position","Level"]]

y=data["Salary"]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor,plot_tree

dt=DecisionTreeRegressor()

dt.fit(x_train,y_train)

y_pred=dt.predict(x_test)

from sklearn import metrics

mse = metrics.mean_squared_error(y_test,y_pred)

mse

r2=metrics.r2_score(y_test,y_pred)

r2

dt.predict([[5,6]])

plt.figure(figsize=(20, 8))

plot_tree(dt, feature_names=x.columns, filled=True)

plt.show()

## Output:

![370564790-3fc9b23a-0fde-4eaf-b0ff-56702f6b88aa](https://github.com/user-attachments/assets/98ac36d0-7a1b-4927-a990-31cf21266f58)

![370564812-2df9f62a-2269-47d0-a24a-b949e079cb36](https://github.com/user-attachments/assets/4ea609f5-32c2-489c-8376-ff69d937ace9)

![370564855-0ff01fe2-6154-46d9-9445-ca2210be7f02](https://github.com/user-attachments/assets/de99d499-ec8b-45af-af23-5b6504207f79)

![370564903-5b5ecc14-e805-4186-acf1-2ecf1ec8b659](https://github.com/user-attachments/assets/b57d6ef6-1677-4fb4-943d-f2c0cd8ab01a)




## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
