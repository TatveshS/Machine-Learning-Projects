import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm #support vector machine
from sklearn.metrics import accuracy_score

diab_dataset = pd.read_csv('diabetes.csv')

# print(diab_dataset.head())
# print(diab_dataset.shape)
# print(diab_dataset.groupby('Outcome').mean())

x = diab_dataset.drop(columns= 'Outcome', axis=1)
y= diab_dataset['Outcome']

# print(x)
# print(y)

# DATA STANDARDIZATION

scaler =StandardScaler()

# scaler.fit(x)

# standardized_data = scaler.transform(x)

standardized_data = scaler.fit_transform(x)


# print(standardized_data)

x = standardized_data

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, stratify=y, random_state=2)

classifier =svm.SVC(kernel= 'linear')

classifier.fit(x_train,y_train)

x_train_prediction = classifier.predict(x_train)
training_data_acc = accuracy_score(x_train_prediction, y_train)

# print(training_data_acc)

x_test_prediction = classifier.predict(x_test)
test_data_acc = accuracy_score(x_test_prediction, y_test)

print(test_data_acc)


input_data=(1,103,30,38,83,43.3,0.183,33)

num_array_input_data = np.asarray(input_data)
num_array_input_data_reshape = num_array_input_data.reshape(1,-1)

std_data = scaler.transform(num_array_input_data_reshape)

prediction = classifier.predict(std_data)

print(prediction)


