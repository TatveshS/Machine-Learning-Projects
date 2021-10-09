from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score



mnist= fetch_openml('mnist_784') 

x, y =mnist['data'], mnist['target']

#print(mnist)
#print(x.shape)
# print(y.shape)
# make them 28 x 28


some_digit = x.get(3600)
# some_digit_image = np.reshape(some_digit, (28,28)) #reshape to plot it
# #  plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")


x_train, x_test = x[60000:], x[: 60000]
y_train, y_test = y[60000:], y[: 60000]

shuffle_index = np.random.permutation(60000)
x_train, y_train = x_train[shuffle_index], y_train[shuffle_index]

         #Creating a 2 detector#

y_train = y_train.astyoe(np.int8)
y_train_2 = (y_train==2)
y_test = y_test.astyoe(np.int8)
y_test_2 = (y_test==2)

clf = LogisticRegression(tol = 0.1)
clf.fit(x_train, y_train_2)

print(clf.predict([some_digit]))

a = cross_val_score(clf, x_train, y_train_2, cv=3, scoring="accuracy")
print(a.mean())