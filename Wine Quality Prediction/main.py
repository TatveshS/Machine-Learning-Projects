import numpy  as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

wine_data = pd.read_csv('winequality-red.csv')

# print(wine_data.head())
# print(wine_data.shape)

# print(wine_data.isnull().sum())

# print(wine_data.describe())

# gr = sns.catplot(x='quality', data=wine_data, kind ='count')
# plt.show()

# pl = plt.figure(figsize=(5,5))
# sns.barplot(x='quality',y='volatile acidity', data= wine_data)
# plt.show()


# pl = plt.figure(figsize=(5,5))
# sns.barplot(x='quality',y='citric acid', data= wine_data)
# plt.show()

correlation = wine_data.corr()

#######          constructing a heatmap to understand a correlation between columns          #######

# plt.figure(figsize=(10,10))
# sns.heatmap(correlation,cbar=True, square=True, fmt='.1f', annot= True,annot_kws={'size':8},cmap='Blues')
# plt.show()




x = wine_data.drop('quality',axis=1)


#####  Label BINARIZATION  #########

y=wine_data['quality'].apply(lambda y_value: 1 if y_value >= 7 else 0 )
print(y)


x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, stratify= y, random_state=2)

model = RandomForestClassifier()
model.fit(x_train,y_train)

x_train_predict = model.predict(x_train)
x_train_predict_acc = accuracy_score(x_train_predict, y_train)
print(x_train_predict_acc)

x_test_prediction = model.predict(x_test)
x_test_prediction_acc = accuracy_score(x_test_prediction, y_test)
print(x_test_prediction_acc)


#predictive system
input_data = (8.5,0.28,0.56,1.8,0.092,35.0,103.0,0.9969,3.3,0.75,10.5)
input_data = np.asarray(input_data)
input_data = input_data.reshape(1,-1)

if model.predict(input_data) == 1:
    print("Nice Quality")
else:
    print("Not so good Quality")    

  