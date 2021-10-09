from matplotlib import colors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn import metrics

car_data = pd.read_csv('car data.csv')

# print(car_data.head())
# print(car_data.shape)

# print(car_data.info())

# print(car_data['Fuel_Type'].value_counts())
# print(car_data['Seller_Type'].value_counts())
# print(car_data['Transmission'].value_counts())

car_data.replace({'Fuel_Type':{'Petrol':0, 'Diesel':1,'CNG':2}},inplace=True)
car_data.replace({'Seller_Type':{'Dealer':0, 'Individual':1}},inplace=True)
car_data.replace({'Transmission':{'Manual':0, 'Automatic':1}},inplace=True)

# print(car_data.head())

x = car_data.drop(columns=['Car_Name', 'Selling_Price'], axis=1)
y = car_data['Selling_Price']

x_train,x_test,y_train, y_test = train_test_split(x,y,test_size=0.1,random_state=2)

model_1 = LinearRegression()

model_1.fit(x_train,y_train)

x_train_pred = model_1.predict(x_train)
error_score = metrics.r2_score(x_train_pred, y_train)
print("R squarred error for trained data:", error_score)

#we use accuracy score only in classification problem

# plt.scatter(y_train, x_train_pred)
# plt.xlabel("Actual Price")
# plt.ylabel("Predicted Price")
# plt.title("Actual vs Predicted Price")
# plt.show()

x_test_pred = model_1.predict(x_test)
error_score_2 = metrics.r2_score(x_test_pred, y_test)
print("R squarred error for test data:", error_score_2)

# plt.scatter(y_test, x_test_pred)
# plt.xlabel("Actual Price")
# plt.ylabel("Predicted Price")
# plt.title("Actual vs Predicted Price")
# plt.show()


########           LASSO REGRESSION      ###########


model_2 = Lasso()

model_2.fit(x_train,y_train)
x_train_pred_lasso = model_2.predict(x_train)
error_score_lasso = metrics.r2_score(x_train_pred_lasso, y_train)
print("R squarred error for trained data using Lasso:", error_score_lasso)
# plt.scatter(y_train, x_train_pred_lasso)
# plt.xlabel("Actual Price")
# plt.ylabel("Predicted Price")
# plt.title("Actual vs Predicted Price")
# plt.show()

x_test_pred_lasso = model_2.predict(x_test)
error_score_2_lasso = metrics.r2_score(x_test_pred_lasso, y_test)
print("R squarred error for test data using Lasso:", error_score_2_lasso)
plt.scatter(y_test, x_test_pred_lasso)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.show()


