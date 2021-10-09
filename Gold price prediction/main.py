import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
import matplotlib.pyplot as plt

gold_data = pd.read_csv("gld_price_data.csv")

# print(gold_data.head())
# print(gold_data.shape)

# print(gold_data.tail)

# print(gold_data.info())
# print(gold_data.isnull().sum())

# print(gold_data.describe())

correlation = gold_data.corr()

# plt.figure(figsize=(8,8))
# sns.heatmap(correlation, cbar=True, square=True, fmt='.1f', annot=True, annot_kws={'size':8}, cmap='Blues')
# plt.show()


# print(correlation['GLD'])

#######             checking the distribution    ###########

# sns.distplot(gold_data['GLD'],color='green')
# plt.show()


x= gold_data.drop(columns=['Date','GLD'],axis=1)
y= gold_data['GLD']


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=2)

model = RandomForestRegressor()

model.fit(x_train,y_train)

x_train_pred = model.predict(x_train)
x_train_pred_acc = metrics.r2_score(y_train, x_train_pred)
print("Trained Error square:",x_train_pred_acc)


x_test_pred = model.predict(x_test)
x_test_pred_acc = metrics.r2_score(y_test, x_test_pred)
print("Test Error square:",x_test_pred_acc)
y_test= list(y_test)

plt.plot(y_test,color='blue',label='Actual Value')
plt.plot(x_test_pred,color='red',label='Predicted Vlaue')
plt.title('Actual vs Predicted')
plt.xlabel('Number pf values')
plt.ylabel('Gold Price')
plt.legend()
plt.show()