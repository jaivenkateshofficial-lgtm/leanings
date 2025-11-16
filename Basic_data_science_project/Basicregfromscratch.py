import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv(r"C:\Users\jaivenkateg\Desktop\practice\data_sets\Algerian_forest_fires_dataset_UPDATE.csv",header=1)
print(df[df.isnull().any(axis=1)])
df.drop(index=[122,167],inplace=True)
df[~df['year'].str.isnumeric()]#similarly check for all numeric value


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
df.reset_index()
obj=[feature for feature in df.columns if df[feature].dtype=='O']
for i in obj:
    if i!='Classes':
        df[i]=df[i].astype(float)
df['Classes']=df['Classes'].str.strip()
df['Classes'].unique()
df['Classes']=np.where(df['Classes']=='not fire',0,1)
df['Region']=df['Region'].astype(int)
df['Classes'].value_counts()
plt.style.use('seaborn-v0_8-whitegrid')
df.hist(bins=50,figsize=(20,15))
plt.show()
percentage=df['Classes'].value_counts(normalize=True)*100
plt.pie(percentage,colors=["red",'green'])
plt.show()
df.to_csv(r"C:\Users\jaivenkateg\Desktop\practice\data_sets\Agerian_forest_project_cleaned.csn")
'''
1.can check the box plot to see the outlayers
2.can check heat map see the correlation
'''
df_temp= df[df['Region'] == 1]
plt.subplots(figsize=(13,6))
sns.set_style('whitegrid')
sns.countplot(x='month',hue='Classes',data=df)
plt.ylabel("Number of fires",weight='bold')
plt.xlabel('Months')
plt.title('Fire Analysis of sidi Bell region')