import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
from sklearn import svm
from sklearn.metrics import accuracy_score
import matplotlib. pyplot as plt

loan_data = pd.read_csv('Loan_status.csv')

# print(loan_data.shape)
# print(loan_data.head())

# print(loan_data.isnull().sum())

loan_data = loan_data.dropna()

#label encoding
loan_data.replace({'Loan_Status':{'N':0, 'Y':1}},inplace=True) #pandas method of replacing

# print(loan_data['Loan_Status'])

# print(loan_data['Dependents'].value_counts())

#replacing 3+ with 4
loan_data = loan_data.replace(to_replace='3+', value=4)

# print(loan_data['Dependents'].value_counts())

# data = sns.countplot(x='Education', hue='Loan_Status', data= loan_data)
# plt.show()


# data = sns.countplot(x='Married', hue='Loan_Status', data= loan_data)
# plt.show()


#converting categorical colums to numerical values

loan_data.replace({'Married':{'No':0, 'Yes':1},'Gender':{'Male':1,'Female':0},'Self_Employed':{'Yes':1,'No':0,},'Property_Area':{'Rural':0,'Semiurban':1,'Urban':2},
                    'Education':{'Graduate':1, 'Not Graduate':0}},inplace=True)

# print(loan_data.head())

x= loan_data.drop(columns=['Loan_ID','Loan_Status'],axis=1)
y= loan_data['Loan_Status']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.1, stratify=y, random_state=2)

model = svm.SVC(kernel='linear')

model.fit(x_train,y_train)

training_data_predict = model.predict(x_train)
training_data_predict_acc = accuracy_score(training_data_predict, y_train)
print(training_data_predict_acc)


test_data_predict = model.predict(x_test)
test_data_predict_acc = accuracy_score(test_data_predict, y_test)
print(test_data_predict_acc)



