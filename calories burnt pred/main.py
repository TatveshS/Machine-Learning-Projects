import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBRegressor
from sklearn import metrics
from sklearn.model_selection import train_test_split

cal_data = pd.read_csv('calories.csv')

exe_data = pd.read_csv('exercise.csv')
# print(exe_data.head())
# print(exe_data.shape)
# print(exe_data.info())

# print(exe_data.isnull().sum())

calories_data = pd.concat([exe_data, cal_data['Calories']], axis=1)
# print(calories_data.head())
# print(calories_data.shape)
# print(calories_data.isnull().sum())


sns.set()
plt.figure(figsize=(8,8))
# sns.countplot(calories_data['Gender'])
# plt.show()

# sns.distplot([calories_data['Age']])
# plt.show()

# sns.distplot([calories_data['Height']])
# plt.show()


# sns.distplot([calories_data['Weight']])
# plt.show()

correlation = calories_data.corr()

# sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')
# plt.show()


calories_data.replace({'Gender':{'female':0, 'male':1}}, inplace=True)
# print(calories_data.info())

x= calories_data.drop(columns=['User_ID','Calories'], axis=1)
y = calories_data['Calories']

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=2)

model = XGBRegressor()

model.fit(x_train,y_train)

x_train_pred = model.predict(x_train)
x_train_pred_acc = metrics.mean_absolute_error(y_train, x_train_pred)
print(x_train_pred_acc)

x_test_pred = model.predict(x_test)
x_test_pred_acc = metrics.mean_absolute_error(y_test, x_test_pred)
print(x_test_pred_acc)

input_data = (0,27,171.0,65.0,4.0,85.0,38.6)
input_data = np.asarray(input_data)
input_data = input_data.reshape(1,-1)
pred = model.predict(input_data)
print(pred)