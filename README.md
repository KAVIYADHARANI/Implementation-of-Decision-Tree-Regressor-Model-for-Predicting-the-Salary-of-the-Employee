# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the libraries and read the data frame using pandas.

2. Calculate the null values present in the dataset and apply label encoder.

3. Determine test and training data set and apply decison tree regression in dataset.

4. Calculate Mean square error,data prediction and r2.


## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Kaviya D
RegisterNumber:  212223040089
*/
```
```
import pandas as pd
from sklearn.tree import  plot_tree
data=pd.read_csv("/content/Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x=data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
from sklearn import metrics
mse=metrics.mean_squared_error(y_test,y_pred)
mse
r2=metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])
plt.figure(figsize=(20,8))
plot_tree(dt,feature_names=x.columns,filled=True)
plt.show()
```
## Output:
## Head:
![318851465-1a86df38-3b03-4d75-a5fb-622493747e8e](https://github.com/user-attachments/assets/93d06783-1b85-45f1-be51-540305ac568a)

## Mean Squared Error:
![318851624-66ebd06a-b135-4228-aef3-b5d4e4b34e23](https://github.com/user-attachments/assets/99f11628-9360-4246-8f2e-8a64f2a9342f)


## Predicted Value:
![318851889-997b45a7-45c1-4466-9ea4-48bbd0aae7c7](https://github.com/user-attachments/assets/0873c340-2c3b-4c90-a3f3-2a6018c532c4)

## Decision Tree:

![318851979-bf3388fa-6405-4f98-9320-d24c8dfd537c](https://github.com/user-attachments/assets/11902ca7-77a5-4970-bc1b-2108dc35b6ab)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
