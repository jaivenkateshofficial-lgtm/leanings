import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.linear_model import Ridge,Lasso,LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.model_selection import RandomizedSearchCV
df=pd.read_csv(r'data_sets\cardekho_imputated.csv')
print(df.isnull().sum())
print(df.info())
df.drop(['Unnamed: 0'],axis=1,inplace=True)
numerical_column=[]
catagorical_column=[]
for column in df.columns:
    if(df[column].dtype in ['int64', 'float64']):
        numerical_column.append(column)
    else:
        catagorical_column.append(column)
print(f'numerical_column:{numerical_column},catagorical_column{catagorical_column}')
# Refining catogorical column
df.drop(['car_name','brand'],axis=1,inplace=True)#The model of the car tells you all detials no need of carname and brand
catagorical_column.remove('car_name')
catagorical_column.remove('brand')
for column in catagorical_column:
    print(f'{column}:{df[column].unique()}')
# From this concluded there is reapted data in catogorical feature
# Refaining the numerical feature
for colum in numerical_column:
    if ((df[column] % 1 == 0).all()) and (df.columns.dtype == 'float64'):
        df[column]=df[column].astype(int)

x=df.drop(['selling_price'],axis=1)
y=df['selling_price']

oh=OneHotEncoder(drop='first')
oh2=OneHotEncoder(drop='first')
oh3=OneHotEncoder(drop='first')
oh4=OneHotEncoder(drop='first')
la=LabelEncoder()
sc=StandardScaler()
x['model']=la.fit_transform(x['model'])
print(x['seller_type'].unique())
x['seller_type']=oh4.fit_transform(pd.DataFrame(x['seller_type'])).toarray()
x['fuel_type']=oh2.fit_transform(pd.DataFrame(x['fuel_type'])).toarray()
x['transmission_type']=oh3.fit_transform(pd.DataFrame(x['transmission_type'])).toarray()
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=10)

models={
    'liniarregression':LinearRegression(),
    'lasso':Lasso(),
    'ridge':Ridge(),
    'desiontree':DecisionTreeRegressor(),
    'randomforest':RandomForestRegressor()
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

# From this we observes that random forest is performing well so we nee to hyper parameter tunning for that
params={
    'n_estimators':[90,100,110,120],
    'max_depth':[5,7,8,9,None],
    'max_features':[5,7,'auto',8]
}
rf=RandomForestRegressor()
rc=RandomizedSearchCV(estimator=rf,param_distributions=params,n_iter=100,cv=3,n_jobs=1)
rc.fit(x_train,y_train)

y_train_pred=rc.predict(x_train)
y_test_pred=rc.predict(x_test)

train_mse=mean_squared_error(y_train,y_train_pred)
train_mae=mean_absolute_error(y_train,y_train_pred)
train_r2=r2_score(y_train,y_train_pred)
train_adj_r2=np.sqrt(train_r2)


print(f'train_mae:{train_mae},train_mse:{train_mse},train_r2:{train_r2},train_adj_r2:{train_adj_r2}')

test_mse=mean_squared_error(y_test,y_test_pred)
test_mae=mean_absolute_error(y_test,y_test_pred)
test_r2=r2_score(y_test,y_test_pred)
test_adj_r2=np.sqrt(test_r2)
print(f'test_mae:{test_mae},test_mse:{test_mse},test_r2:{test_r2},test_adj_r2:{test_adj_r2}')

# Need torefine the code some input cauisng issue it worked while running for the firest time in debugiing 
# check the code and make it proper