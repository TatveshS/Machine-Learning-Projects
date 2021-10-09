import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

customer_data = pd.read_csv('Mall_Customers.csv')

# print(customer_data.head())
# print(customer_data.shape)
# print(customer_data.info())
# print(customer_data.isnull().sum())


x = customer_data.iloc[:,[3,4]].values
# print(x)

#### CHOOSING THE NUMBER OF CLUSTERS
# WCSS --> Within cluster sum of squares

   #ELBOW METHOD

    # finding minimum wcss value for different number of clusters
        
wcss = []

for i in range(1,11):
    kmeans= KMeans(n_clusters=i, init='k-means++',random_state=42)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

   #plotting an elbow graph

# sns.set()
# plt.plot(range(1,11),wcss)
# plt.title('The Elbow point graph')
# plt.xlabel('Number of clusters')
# plt.ylabel('WCSS')
# plt.show()  # CHECK FOR SIGNIFICANT DROP IN THE VALUE (HERE IT IS 3 AND 5, WE WILL CONSIDER 5 AS THERE IS VERY LESS CHANGE AFTER THAT)

# optimum number of clusters = 5

###### TRAINING THE MODEL #######
kmeans = KMeans(n_clusters=5, init='k-means++',random_state=0)

# return a label for each data point based on their cluster

y = kmeans.fit_predict(x)
# print(y)


##### VISUALIZING ALL THE CLUSTERS
plt.figure(figsize=(8,8))
plt.scatter(x[y==0,0], x[y==0,1],s=50,c='green', label= 'Cluster 1')
plt.scatter(x[y==1,0], x[y==1,1],s=50,c='red', label= 'Cluster 2')
plt.scatter(x[y==2,0], x[y==2,1],s=50,c='yellow', label= 'Cluster 3')
plt.scatter(x[y==3,0], x[y==3,1],s=50,c='violet', label= 'Cluster 4')
plt.scatter(x[y==4,0], x[y==4,1],s=50,c='blue', label= 'Cluster 5')

#plot the centroids
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],s=100,c='cyan',label='centroid')
plt.title('Customer Groups')
plt.xlabel('Annual Income')
plt.ylabel('Spending Score')


plt.show()