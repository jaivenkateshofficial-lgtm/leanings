import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt 
import scipy.cluster.hierarchy as sc
from sklearn.cluster import AgglomerativeClustering

iris=load_iris()
df=pd.DataFrame(iris.data,columns=iris.feature_names)
scalar=StandardScaler()
df_scaled=scalar.fit_transform(df)#we need to calculate the distance from points so all points should not be seperate distance
pca=PCA(n_components=2)
df_pca=pca.fit_transform(df_scaled)
plt.figure(figsize=(10,6))
plt.scatter(df_pca[:,0],df_pca[:,1],c=iris.target)

# Agormative dendogram
sc.dendrogram(sc.linkage(df_pca,method='ward'))
plt.title('dendogram')
plt.xlabel('sample index')
plt.ylabel('Eculadian distance')

# Find the logest vertical line and no  horizotal line not passes through it.
ag=AgglomerativeClustering(n_clusters=2,metric='euclidean',linkage='ward')
ag.fit(df_pca)
plt.scatter(df_pca[:,0],df_pca[:,1],c=ag.labels_)