#RE PRICE PR
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import standard_t, triangular
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load


housing = pd.read_csv(r'C:\Users\Tatvesh Sawant\Desktop\PYTHON PROJECTS\Machine Learning\RealEstate\data.csv')
#print(housing.head())
#print(housing.info())
#print(housing['CHAS'].value_counts())
#print(housing.describe())


#housing.hist(bins=50, figsize=(20,15))
#plt.show()


#train-test splitting (Present in sklearn)
# def split_train_test(data, test_ratio):
#     np.random.seed(42)#to provide overfiiting it shuffles the data just one time
#     shuffled= np.random.permutation(len(data))
#     test_set_size = int(len(data)*test_ratio)
#     test_indices = shuffled[:test_set_size]
#     train_indices = shuffled[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices] #ilocgives rows

# train_set, test_set = split_train_test(housing,0.2)
# print(f"Rows in train set:{len(train_set)} \nRows in train set:{len(test_set)}")

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
print(f"Rows in train set:{len(train_set)} \nRows in test set:{len(test_set)}")

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):#splitting housing wrt housing['CHAS']
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

housing = strat_train_set.copy()

#Looking for Correlations (this is done for removing outline points like rm=4 and rm=9 has same price)
corr_matrix = housing.corr()
print(corr_matrix['MEDV'].sort_values(ascending=False))#MEDV is price in file

attributes=['MEDV','ZN','RM','LSTAT']
#scatter_matrix(housing[attributes], figsize=[12,8])
housing.plot(kind ='scatter', x='RM',y='MEDV')
plt.show()

housing = strat_train_set.drop('MEDV', axis=1)
housing_labels= strat_train_set['MEDV'].copy()

#ATTRIBUTE COMBINATION
#housing['TAXRM'] = housing['TAX']/housing['RM']

# Taking care of missing attributes
#   1.getting rid of missing data points
#   2.get rid of whole attributes
#   3.set the value to some value (0, mean or median)

# a = housing.dropna(subset=['RM'])#option 1
# print(a.shape)
# b = housing.drop('RM', axis=1)#option2
# print(b.shape)

# median = housing['RM'].median() #option3
# housing['RM'].fillna(median)

imputer = SimpleImputer(strategy= "median") #does the job of option3 effectively i.e for test data also
imputer.fit(housing) #fits median for every column
print(imputer.statistics_)

X  = imputer.transform(housing)
housing_tr = pd.DataFrame(X, columns= housing.columns)
print(housing_tr.describe())

##feature scaling(Feature Scaling is a technique to standardize the independent features present in the data in a fixed range. It is performed during the data pre-processing to handle highly varying magnitudes or values or units.)
    #   1.Min-max scaling(Normalization)
    #   (value-min)/(max-min)
    #   sklearn minmaxscalar

    #   2. Standardization
    #   (value-min)/standard deviation
    #   sklearn standar scaler


##CREATING PIPELINE
my_pipeline = Pipeline([

    ('imputer', SimpleImputer(strategy='median')),#add as many as you want
    ('std_scaler', StandardScaler())

    ])

housing_num_tr = my_pipeline.fit_transform(housing_tr)
print(housing_num_tr)#it gives numpy array because the scikit predictor acts on arrays


 
 
                ##          SELECTING A DESIRED MODEL FOR DRAGON REAL ESTATES            ##

#model = LinearRegression() 
#model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)

some_data = housing[:5]
some_labes = housing_labels[:5]
prepared_data = my_pipeline.transform(some_data)
model.predict(prepared_data)

print(list(some_labes))

##EVALUATING the model
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = np.sqrt(mse)

print(rmse) #if this gives 0 then it is OVERFITTING

#USING BETTER EVALUATION TECHNIQUE  - CROSS VALIDATION
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring = "neg_mean_squared_error", cv=10)#cv is folds
rmse_scores = np.sqrt(-scores)
print(rmse_scores)

def print_scores(scores):
    print("Scores: ", scores)
    print("Mean: ", scores.mean())
    print("Standard deviation: ", scores.std())

print(print_scores(rmse))    


#SAVING THE MODEL
dump(model, 'Dragon.joblib')



           ##TESTING THE DATA

x_test = strat_test_set.drop('MEDV', axis=1)
y_test = strat_test_set['MEDV']
x_test_prepared = my_pipeline.transform(x_test)
final_predictions= model.predict(x_test_prepared)
final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
print(final_rmse)

