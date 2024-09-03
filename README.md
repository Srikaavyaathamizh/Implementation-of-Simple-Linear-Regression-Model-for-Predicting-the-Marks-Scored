# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary libraries for data handling, visualization, and model building
2. Load the dataset and inspect the first and last few records to understand the data structure.
3.Prepare the data by separating the independent variable (hours studied) and the dependent variable (marks scored).
4. Split the dataset into training and testing sets to evaluate the model's performance.
5.Initialize and train a linear regression model using the training data.
6.Predict the marks for the test set using the trained model.
7.Evaluate the model by comparing the predicted marks with the actual marks from the test set.
8.Visualize the results for both the training and test sets by plotting the actual data points and the regression line


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SRIKAAVYAA T
RegisterNumber:  212223230214
*/
import numpy as np
import pandas as pd
from sklearn.metrics import  mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
dataset = pd.read_csv('student_scores.csv')
print(dataset.head())
print(dataset.tail())
```

## Output:
![Screenshot 2024-09-03 134738](https://github.com/user-attachments/assets/26358b76-eaa7-4be1-8e34-a907ecdb7b1e)

```
dataset.info()
```

![Screenshot 2024-09-03 134848](https://github.com/user-attachments/assets/dc72c15a-3c7b-4560-a868-8c328f610ebf)
```
X=dataset.iloc[:,:-1].values
print(X)
Y=dataset.iloc[:,-1].values
print(Y)
```
##  OUTPUT

![Screenshot 2024-09-03 135021](https://github.com/user-attachments/assets/c795e674-e5a9-44db-adb7-1a993e5a4977)
```
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=1/3,random_state=0)
print(X_train.shape)
print(X_test.shape)
```
## OUTPUT

![Screenshot 2024-09-03 135129](https://github.com/user-attachments/assets/5d313a26-47bd-4f5d-938b-ee6d9bf3f2c4)

```
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(X_train,Y_train)
```
## OUTPUT

![Screenshot 2024-09-03 135309](https://github.com/user-attachments/assets/976fea73-78cb-499f-ab82-e3511a657470)

```
Y_pred=reg.predict(X_test)
print(Y_pred)
print(Y_test)
```
## OUTPUT


![Screenshot 2024-09-03 135615](https://github.com/user-attachments/assets/3f966d83-70d4-47f8-9ebb-d2edbb651f44)
```
plt.scatter(X_train,Y_train,color="red")
plt.plot(X_train,reg.predict(X_train),color="green")
plt.title('Training set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```

![Screenshot 2024-09-03 140330](https://github.com/user-attachments/assets/79713e5e-817e-42cb-8c3a-7ca058403a82)

```
plt.scatter(X_test, Y_test,color="blue")
plt.plot(X_test, reg.predict(X_test), color="silver")
plt.title('Test set(H vs S)')
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
```


![Screenshot 2024-09-03 140516](https://github.com/user-attachments/assets/6f07fef9-60a2-472c-90e5-573f22367f62)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
