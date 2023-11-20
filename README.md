# Ex no: 8 Implementation-of-SVM-For-Spam-Mail-Detection
date: 9/11/23
## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary python packages using import statements.
2. Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().
3. Split the dataset using train_test_split.
4. Calculate Y_Pred and accuracy.
5. Print all the outputs.
6. End the Program.


## Program:

Program to implement the SVM For Spam Mail Detection..
Developed by: Bala Umesh
RegisterNumber:  212221040024

``` py
import pandas as pd
data=pd.read_csv("spam.csv",encoding='latin-1')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extractiaon.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)

y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
### data.head():
![image](https://github.com/BalaUmesh/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113031742/3f128ae4-cb08-4db1-8fe9-98f8adf4e888)


### data.info():
![image](https://github.com/BalaUmesh/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113031742/4de1f123-cfb1-4065-88ef-119155b9fa69)


### data.isnull().sum():
![image](https://github.com/BalaUmesh/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113031742/d614a98b-3691-4ed1-918b-169956d6ec9e)


### y-pred:
![image](https://github.com/BalaUmesh/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113031742/f59a2395-0f22-4cbd-9405-3f60088bfd49)


### accuracy:
![image](https://github.com/BalaUmesh/Implementation-of-SVM-For-Spam-Mail-Detection/assets/113031742/00b72d44-7ede-488d-b3ab-c1503f409a3f)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
