# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.

2.Set variables for assigning dataset values.

3.Import linear regression from sklearn.

4.Assign the points for representing in the graph.

5.Predict the regression for marks by using the representation of the graph.

6.Compare the graphs and hence we obtained the linear regression for the given datas.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Kannadhasan J
RegisterNumber: 212224240071 
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x = df.iloc[:,:-1].values
print(x)
y = df.iloc[:,1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
print(y_pred)
print(y_test)
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='black')
plt.plot(x_train,regressor.predict(x_train),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

## Output:
### DATASET:
![m1](https://github.com/user-attachments/assets/ea243a12-4b92-4799-82db-43b20ec96cbd)
### HEAD VALUES:
![m2](https://github.com/user-attachments/assets/a3fad7eb-a74f-45f7-900e-ed9639cfc7a4)
### TAIL VALUES:
![m3](https://github.com/user-attachments/assets/91a2caa8-bbb9-4859-96ab-39e519ba623b)

### X AND Y VALUES:
![m4](https://github.com/user-attachments/assets/292d3e7c-c270-496c-8dbc-ff7119648813)

### PREDICTION VALUES:
![m5](https://github.com/user-attachments/assets/2e5d3f0c-2c74-40d4-9004-f26e3660c63b)

### TRAINING SET:
![m6](https://github.com/user-attachments/assets/a5a744e7-5105-438d-9aac-c724834730d7)

### TESTING SET:
![m7](https://github.com/user-attachments/assets/e94c365d-84b8-4cc1-8a7e-504a745a2929)

### MSE,MAE AND RMSE:
![m8](https://github.com/user-attachments/assets/4070ef45-c6b4-496a-bd9d-d66fc5b55178)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
