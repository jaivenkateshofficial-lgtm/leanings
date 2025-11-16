import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r'C:\Users\jaivenkateg\Desktop\practice\data_sets\winequality-red.csv')
print(df.info())#summary of data set
print(df.columns)
print(df['alcohol'].unique())
pl=df['alcohol'].unique()
print(df.isnull().sum())
print(df[df.duplicated])
print(df.drop_duplicates(inplace=False))#if you inplace true it will permently remove values
print(df.corr())
# sns.heatmap(df.corr(),annot=True)
plt.figure(figsize=(10,6))
# sns.scatterplot(x=df['alcohol'],y=df['pH'])
sns.barplot(x=df['citric acid'],y=df['pH'])
plt.show()

