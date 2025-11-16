import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import statsmodels.api as sm

df=pd.read_csv(r'C:\Users\jaivenkateg\Desktop\practice\data_sets\economic_index.csv')
df.info()
df.drop(columns=['month','year','Unnamed: 0'],axis=1,inplace=True)
df.info()
print(df.isnull().sum())
sns.pairplot(df)
df.corr()
x=df[['interest_rate','unemployment_rate']]#df.iloc[:,:-1]
y=df.iloc[:,-1]#other method
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
sns.regplot(x=df['interest_rate'],y=df['index_price'])
scalar=StandardScaler()
x_train=scalar.fit_transform(x_train)
x_test=scalar.fit_transform(x_test)
reg=LinearRegression()
reg.fit(x_train,y_train)
val_score=cross_val_score(reg,x_train,y_train,scoring='neg_mean_squared_error',cv=3)
np.mean(val_score)
# Predict
y_pred=reg.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
score=r2_score(y_test,y_pred)

# Assemptions
plt.scatter(y_test,y_pred)
residuals=y_test-y_pred
sns.displot(residuals,kind='kde')
plt.scatter(y_pred,residuals)
model=sm.OLS(y_train,x_train).fit()
model.summary()
print(reg.coef_)