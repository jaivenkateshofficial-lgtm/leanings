import pandas as pd

df=pd.read_csv('data.csv')
# print(df.head(5))
# print(df.tail(5))
# print(df.describe)
# handling missing value
print(df.isnull())
# print(df.isnull.axis(1))
print(df.isnull.sum())