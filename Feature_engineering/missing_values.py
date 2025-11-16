import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plot
df=sns.load_dataset("titanic")
print(df.dropna(axis=1,))
print(df.isnull().sum())
print(sns.displot(df['age']))
df['Age_mean']=df['age'].fillna(df['age'].mean())
df.shape()
print(df[['Age_mean','age']])
plot.show()
# Mean imputation works well when we have normally distributed data
# Median value imputatio technique when we have oulayers in 
# use when catogical value