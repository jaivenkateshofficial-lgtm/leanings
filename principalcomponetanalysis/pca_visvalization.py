import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

cancer_dataset=load_breast_cancer()
df=pd.DataFrame(cancer_dataset['data'],columns=cancer_dataset['feature_names'])
scalar=StandardScaler()
scalar.fit(df)
scaled_data=scalar.transform(df)
pca=PCA(n_components=2)
data_pca=pca.fit_transform(scaled_data)
plt.figure(figsize=(8,6))
plt.scatter(data_pca[:,0],data_pca[:,1],cancer_dataset['target'],cmap='plasma',edgecolors='k')
plt.xlabel("First principle componet")
plt.ylabel('Second principle componet')