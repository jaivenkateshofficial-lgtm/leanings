import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv(r"https://raw.githubusercontent.com/krishnaik06/playstore-Dataset/main/googleplaystore.csv")
# print(df.head())
# print(df.info())
# print(df.shape)
# print(df.describe())
print(df.isnull().sum())
# Insight:
# The data set has an missing values
# print(df.head(3))
# df['Reviews'].unique()[:8].tolist()#toget the full data
# print(df['Rating'].unique().tolist())
# print(df['Size'].unique().tolist())
pd.set_option('display.max_colwidth', None) 
print(df['Reviews'].str.isnumeric().sum())
print(df[~df['Reviews'].str.isnumeric()])
df_work=df.copy()
df_work=df.drop(index=10472)

print(df_work.info())
df_work['Reviews']=df_work['Reviews'].astype(int)
df_work['Size'].str.isnumeric().sum()#To find all having M or K
df_work['Size'].unique()#to find all the unique value
df_work['Size']=df_work['Size'].str.replace('M','000')
df_work['Size']=df_work['Size'].str.replace('k','')
df_work['Size']=df_work['Size'].replace('Varies with device',np.nan)
remove_element=['+',",",'$']
col_clean=['Installs','Price']
for item in remove_element:
    for col in col_clean:
        df_work[col]=df_work[col].str.replace(item,'')

df_work['Price']=df_work['Price'].astype(float)
df_work['Installs']=df_work['Installs'].astype(float)
df_work['Last Updated']=pd.to_datetime(df_work['Last Updated'])
df_work['Day']=df_work['Last Updated'].dt.day
df_work['Month']=df_work['Last Updated'].dt.month
df_work['Year']=df_work['Last Updated'].dt.year
df_work.drop('Last Updated',axis=1)
df_work.to_csv(r"C:\Users\jaivenkateg\Desktop\practice\data_sets\gogle_cleaned_data.csv")
df_work.to_excel(r"C:\Users\jaivenkateg\Desktop\practice\data_sets\gogle_cleaned_data.xlsx")
df_work=df_work.drop_duplicates(subset=['App'],keep='first')
'''
Exploratary Data Analysi:

Independent and depdent catogories
weight is independent and height is dependent
x=df[[]]-to get a data frame ,df[]-to get list 
Always you're indpendent feature should be data frame or 2d array
Train Test split
'''
df_work[df.duplicated('App')].shape
# Infrance: there are duplicate recods
df_work = df_work.drop_duplicates(subset=['App'], keep='first')
numerical_features=[feature for feature in df_work.columns if df_work[feature].dtype !='O' ]#The featurer which are numerical either int or float
catogorical_value=[feature for feature in df_work.columns if df_work[feature].dtype =='O' ]#Features which are catogorical
print(f'The numeric feature are{numerical_features} ,The Catogorical feature{catogorical_value}')
for col in catogorical_value:
    print(df[col].value_counts(normalize=True)*100)
'''
Understanding Distribution: By using value_counts(normalize=True), you determine how many times each category appears in the column, expressed as a proportion of the total count. This is essential for understanding the distribution of categorical data.
Percent Visualization:
normalize=True, the output will show the proportion of each category as a fraction of the total counts. This is beneficial for understanding how categories compare to one another in terms of share or significance rather than just raw numbers.
kd-kurnel desity estimate
'''
# Propotional count of numerical columns
plt.figure(figsize=(15,15))
plt.suptitle('universal analysis of numeric features',fontsize=20,fontweight='bold',alpha=0.8,y=1)
for i in range(0,len(numerical_features)):
    plt.subplot(5,3,i+1)
    sns.countplot(x=df_work[catogorical_value[i]],palette='Set2')
    sns.kdeplot(x=df_work[numerical_features[i]],fill=True,color='r')
    plt.xlabel(numerical_features[i])
    plt.tight_layout()
df['Category'].value_counts().plot.pie()
plt.figure(figsize=(10, 6))
print('hello')
