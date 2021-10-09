import numpy as np
import pandas as pd
from pandas.core.algorithms import mode
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sonar_data= pd.read_csv('Copy of sonar data.csv',header=None)

#print(sonar_data.head())

# print(sonar_data.shape)

print(sonar_data[60].value_counts())

x= sonar_data.drop(columns=60, axis=1)
y= sonar_data[60]

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1, stratify=y, random_state=1)

print(x.shape, x_test.shape, x_train.shape)

model = LogisticRegression()

model.fit(x_train,y_train)

x_train_predict = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_predict, y_train)

print("Accuracy of Training data:", training_data_accuracy)

x_test_predict = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_predict, y_test)

print("Test Data Accuracy:", test_data_accuracy)


#MAKING A PEDICTIVE SYSTEM

input_data=(0.0124,0.0433,0.0604,0.0449,0.0597,0.0355,0.0531,0.0343,0.1052,0.2120,0.1640,0.1901,0.3026,0.2019,0.0592,0.2390,0.3657,0.3809,0.5929,0.6299,0.5801,0.4574,0.4449,0.3691,0.6446,0.8940,0.8978,0.4980,0.3333,0.2350,0.1553,0.3666,0.4340,0.3082,0.3024,0.4109,0.5501,0.4129,0.5499,0.5018,0.3132,0.2802,0.2351,0.2298,0.1155,0.0724,0.0621,0.0318,0.0450,0.0167,0.0078,0.0083,0.0057,0.0174,0.0188,0.0054,0.0114,0.0196,0.0147,0.0062)
input_data_2=(0.0114,0.0222,0.0269,0.0384,0.1217,0.2062,0.1489,0.0929,0.1350,0.1799,0.2486,0.2973,0.3672,0.4394,0.5258,0.6755,0.7402,0.8284,0.9033,0.9584,1.0000,0.9982,0.8899,0.7493,0.6367,0.6744,0.7207,0.6821,0.5512,0.4789,0.3924,0.2533,0.1089,0.1390,0.2551,0.3301,0.2818,0.2142,0.2266,0.2142,0.2354,0.2871,0.2596,0.1925,0.1256,0.1003,0.0951,0.1210,0.0728,0.0174,0.0213,0.0269,0.0152,0.0257,0.0097,0.0041,0.0050,0.0145,0.0103,0.0025)

#changing the inp data to numpy array
input_data_as_numpy_array = np.asarray(input_data)
input_data_as_numpy_array_2 = np.asarray(input_data_2)

#reshaping for 1 instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
input_data_reshaped_2 = input_data_as_numpy_array_2.reshape(1,-1)

prediction = model.predict(input_data_reshaped)
prediction_2 = model.predict(input_data_reshaped_2)

print(prediction)
print(prediction_2)

