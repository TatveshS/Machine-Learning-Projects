import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

heart_data = pd.read_csv('heart.csv')
# print(heart_data.head())
# print(heart_data.shape)
# print(heart_data.info())

# print(heart_data.isnull().sum())

# print(heart_data['target'].value_counts())

x = heart_data.drop(columns=['target'],axis=1)
y= heart_data['target']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,stratify=y,random_state=2)

model = LogisticRegression()

model.fit(x_train,y_train)

x_train_pred = model.predict(x_train)
x_train_pred_acc = metrics.accuracy_score(x_train_pred,y_train)
print("Training Accuracy:", x_train_pred_acc)


x_test_pred = model.predict(x_test)
x_test_pred_acc = metrics.accuracy_score(x_test_pred,y_test)
print("Training Accuracy:", x_test_pred_acc)


input_data = (49,1,1,130,266,0,1,171,0,0.6,2,0,2)

input_data = np.asarray(input_data)
input_data = input_data.reshape(1,-1)
pred = model.predict(input_data)
if pred[0]== 1:
    print("The patient has heart disease")
else:
    print("The person has no heart disease")    




