import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from kneed import KneeLocator

x,y=make_blobs(n_samples=1000,centers=3,n_features=2)
plt.scatter(x[:,0],x[:,1],c=y)

scalar=StandardScaler()
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=10,test_size=0.33)
x_train_scaled=scalar.fit_transform(x_train)
x_test_scalled=scalar.transform(x_test)

wcss=[]
for k in range(1,12):
    kmeans=KMeans(n_clusters=k,init='k-means++')
    kmeans.fit(x_train_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10,9))
plt.plot(range(1,12),wcss)
plt.xticks(range(1,12))
plt.xlabel('kvalue')
plt.ylabel('wcss')

kl=KneeLocator(range(1,12),wcss,curve='convex', direction='decreasing')
print(kl.elbow)
# silhotscore
s_coeff=[]
for k in range(2,11):
    kmeans=KMeans(n_clusters=k,init='k-means++')
    kmeans.fit(x_train_scaled)
    score=silhouette_score(x_train_scaled,kmeans.labels_)
    s_coeff.append(score)
plt.figure(figsize=(10,9))
plt.plot(range(2,11),s_coeff)
plt.xticks(range(2,11))
plt.xlabel('kvalue')
plt.ylabel('score')

