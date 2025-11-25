import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
from xgboost import XGBRegressor
df=pd.read_csv(r'data_sets\tb.csv',index_col=0)
df.drop('Player',axis=1,inplace=True)
df[['start', 'end']] = df['Span'].str.split('-', expand=True)
df.drop('Span',axis=1,inplace=True)
df[['start','end']]=df[['start','end']].astype(int)
df['toatal years']=df['start']-df['end']
df.drop(['start','end'],axis=1,inplace=True)
df.HS[~df['HS'].str.isnumeric()]#To check the non numeric values
df['HS']=df['HS'].str.strip('*')
df['HS']=df['HS'].astype(int)
df['Mat']=df['Mat'].str.strip('*')
df['Mat']=df['Mat'].astype(int)
df.drop('NO',axis=1,inplace=True) 

# Selscting the independent and dependent variable
x=df.drop(['Runs'],axis=1)
y=df['Runs']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=10)

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

models={
    'Liniarregression':LinearRegression(),
    'Adabostregressor':AdaBoostRegressor(),
    'Gradientboost':GradientBoostingRegressor(),
    'DesionTreeRegressor':DecisionTreeRegressor(),
    'Xgboost':XGBRegressor()
}

for i in range(len(list(models))):
    model=list(models.values())[i]

    model.fit(x_train,y_train)
    y_train_pred=model.predict(x_train)
    y_test_pred=model.predict(x_test)

    train_mse=mean_squared_error(y_train,y_train_pred)
    train_mae=mean_absolute_error(y_train,y_train_pred)
    train_r2=r2_score(y_train,y_train_pred)
    train_adj_r2=np.sqrt(train_r2)

    print(f'The permoace for model{list(models)[i]}')

    print(f'train_mae:{train_mae},train_mse:{train_mse},train_r2:{train_r2},train_adj_r2:{train_adj_r2}')

    test_mse=mean_squared_error(y_test,y_test_pred)
    test_mae=mean_absolute_error(y_test,y_test_pred)
    test_r2=r2_score(y_test,y_test_pred)
    test_adj_r2=np.sqrt(test_r2)

    print(f'test_mae:{test_mae},test_mse:{test_mse},test_r2:{test_r2},test_adj_r2:{test_adj_r2}')

