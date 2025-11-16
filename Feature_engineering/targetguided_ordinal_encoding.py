import pandas as pd
df1=pd.DataFrame({
    'city':['chennai','mumbai','kolkata','Delhi','Banglore','Pune','chennai'],
    'prize':[200,300,190,250,240,230,230]
})
print(df1.groupby('city')['prize'].mean())
mean_prize=df1.groupby('city')['prize'].mean()
df1['city_encoding']=df1['city'].map(mean_prize)
print(df1)
print(df1[['city_encoding','city']])