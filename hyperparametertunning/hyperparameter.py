import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r'C:\Users\jaivenkateg\Desktop\practice\data_sets\Algerian_forest_fires_dataset_UPDATE.csv',header=1)

# Data cleaning
print(df[df.isnull().any(axis=1)])
df.loc[:122,"Region"]=0
df.loc[122:,"Region"]=1
df.isnull().sum()
df.dropna().reset_index()
df.drop(122).reset_index()
# fix spaces in column name
df.columns=df.columns.str.strip()#this will strp off white spaces
# Change to numerical value
df.drop([122,123,167],inplace=True)
df[[ 'day', 'month', 'year', 'Temperature', 'RH','Ws']]=df[[ 'day', 'month', 'year', 'Temperature', 'RH','Ws']].astype(int)
df.drop([11,167],inplace=True)
df.reset_index()
obj=[feature for feature in df.columns if df[feature].dtype=='O']
for i in obj:
    if i!='Classes':
        df[i]=df[i].astype(float)
df.to_csv(r"C:\Users\jaivenkateg\Desktop\practice\data_sets\Algerian_forest_fires_dataset_Cleaned_jai.csv",index=False)#to stop storing it again
df_copy=df.drop(['day', 'month', 'year'],axis=1)
# Encoding
# df_copy['Classes']=np.where(df_copy['Classes']=='not fire',0,1)
df_copy['Classes']=np.where(df_copy['Classes'].str.contains('not fire'),0,1)
df_copy['Classes']=np.where(df_copy['Classes']=='not fire',0,1)
df_copy.hist(bin=50,figsize=(20,15))
# percentage of pie chart
percentage=df_copy['Classes'].value_counts(normalize=True)*100
classlabel=['Fire','Not fire']
plt.figure(figsize=(12,7))
plt.pie(percentage,labels=classlabel,autopct='%1 1.1f%%',colors=['red','Green'])
plt.title('This my pie chart')
sns.heatmap(df_copy.corr())
# Mothly Fire Analysis
dftemp=df.loc[df['Region']==1]
plt.subplots(figsize=(13,6))
sns.set_style('whitegrid')
sns.countplot(x='month',hue='Classes',data=df)