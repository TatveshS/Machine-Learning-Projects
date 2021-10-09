import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

par_data = pd.read_csv('parkinsons.csv')
# print(par_data.head())
# print(par_data.shape)
# print(par_data.info())

# print(par_data.isnull().sum())

# print(par_data.describe())

# print(par_data['status'].value_counts())

# print(par_data.groupby('status').mean())


x = par_data.drop(columns=['name','status'], axis=1)
y = par_data['status']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, stratify=y, random_state=2)

scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) # should not fit test data

model = svm.SVC(kernel='linear')

model.fit(x_train, y_train)

x_train_pred = model.predict(x_train)
x_train_pred_acc = accuracy_score(x_train_pred, y_train)
print(x_train_pred_acc)

x_test_pred = model.predict(x_test)
x_test_pred_acc = accuracy_score(x_test_pred, y_test)
print(x_test_pred_acc)


input_data = (198.38300,215.20300,193.10400,0.00212,0.00001,0.00113,0.00135,0.00339,0.01263,0.11100,0.00640,0.00825,0.00951,0.01919,0.00119,30.77500,0.465946,0.738703,-7.067931,0.175181,1.512275,0.096320)
input_data = np.asarray(input_data)
input_data = input_data.reshape(1,-1)
input_data = scaler.transform(input_data)
pred = model.predict(input_data)

if pred[0]==1:
    print("The person has Parkinsons disease")
else:
    print("The person does not have Parkinsons disease")    