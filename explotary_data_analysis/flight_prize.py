import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder

df=pd.read_excel(r'C:\Users\jaivenkateg\Desktop\practice\data_sets\flight_price.xlsx')
print(df)
print(df.info())
df['date']=df['Date_of_Journey'].str.split('/').str[0]
df['month']=df['Date_of_Journey'].str.split('/').str[1]
df['year']=df['Date_of_Journey'].str.split('/').str[2]
df['date']=df['date'].astype(int)
df['month']=df['month'].astype(int)
df['year']=df['year'].astype(int)
print(df.info())
df.drop('Date_of_Journey', axis=1, inplace=True)
df['Arrival_Time']=df['Arrival_Time'].apply(lambda x:x.split(' ')[0])
df['Arriva_hour']=df['Arrival_Time'].str.split(':').str[0]
df['Arrival_minute']=df['Arrival_Time'].str.split(':').str[1]
df['Arriva_hour']=df['Arriva_hour'].astype(int)
df['Arrival_minute']=df['Arrival_minute'].astype(int)
print(df.info())
print(df.head())
print(df['Total_Stops'].unique())
df['Total_Stops']=df['Total_Stops'].map({'non-stop':0, '1 stop':1,'2 stops':2,'3 stops':3,'4 stops':4})
print(df.head())
print(df['Total_Stops'])
df['Duration_hours']=df['Duration'].str.split(" ").str[0].str.strip("h").str[0]
df['Duration_minutes']=df['Duration'].str.split(" ").str[1].str.strip("m").str[0]
df['Duration_hours']=df['Duration_hours'].astype(int)
df.drop('Duration', axis=1, inplace=True)
# df['Duration_minutes']=df['Duration_minutes'].astype(int)
print(df['Duration_hours'])
encoder=OneHotEncoder()
encode=encoder.fit_transform(df[['Airline','Source','Destination']]).toarray()
df2=pd.DataFrame(encode,columns=encoder.get_feature_names_out())
print(df2)
df.drop(['Airline','Source','Destination'], axis=1, inplace=True)
df=pd.concat([df,df2],axis=1)
print(df.info())