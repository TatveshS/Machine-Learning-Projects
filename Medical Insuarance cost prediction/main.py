import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression


medical_data = pd.read_csv('insurance.csv')

# print(medical_data.shape)
# print(medical_data.head())
# print(medical_data.info())

sns.set()
plt.figure(figsize=(6,6))
# sns.distplot(medical_data['age'])
# plt.title('Age Distribution')
# plt.show()

# sns.countplot(x='sex', data=medical_data)
# plt.title('Sex Distribution')
# plt.show()


# sns.distplot(medical_data['bmi'])
# plt.title('BMI Distribution')
# plt.show()

# sns.countplot(x='children', data=medical_data)
# plt.title('Children Distribution')
# plt.show()


medical_data.replace({'smoker':{'yes':1, 'no':0}},inplace=True)
medical_data.replace({'sex':{'female':1, 'male':0}},inplace=True)
medical_data.replace({'region':{'southeast':0, 'southwest':1, 'northeast':2, 'northwest':3}},inplace=True)

x = medical_data.drop(columns='charges',axis=1)
y = medical_data['charges']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=2)

model = LinearRegression()

model.fit(x_train,y_train)

x_train_pred = model.predict(x_train)
x_train_pred_acc = metrics.r2_score(y_train,x_train_pred)
print(x_train_pred_acc)

x_test_pred = model.predict(x_test)
x_test_pred_acc = metrics.r2_score(y_test,x_test_pred)
print(x_test_pred_acc)


input_data =(56,1,39.82,0,0,0)
input_data = np.asarray(input_data)
input_data = input_data.reshape(1,-1)
pred = model.predict(input_data)

print("Predicted data is: ", pred)