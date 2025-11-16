import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import _california_housing
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import pickle

california=pd.read_csv(r'C:\Users\jaivenkateg\Desktop\practice\data_sets\housing.csv')
df=california
print("hello")
# Devide it into dependent and independent
df1=df.drop(['ocean_proximity'],axis=1)
sns.heatmap(df1.corr(),annot=True)
df1.isnull().sum()
df1.dropna(inplace=True)
x=df1.iloc[:,:-1]
y=df1.iloc[:,-1]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=33,random_state=10)
scalar=StandardScaler()
x_train=scalar.fit_transform(x_train)
x_test=scalar.transform(x_test)
lin=LinearRegression()
lin.fit(x_train,y_train)
lin.coef_
y_pred=lin.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mse)
rscore=r2_score(y_test,y_pred)
print(f'mse{mse},mae{mae},rmse{rmse},r2score{rscore}')
adjscore=1-(1-rscore)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)
'''
y test and ypred
'''
plt.scatter(y_test,y_pred)

'''
2.Residuals:will see 
'''
res=y_test-y_pred
sns.displot(res,kde=True)

'''
3.scatter plot with respect to predection and residuals
This should be an uniform distribution
'''
plt.scatter(y_pred,res)
pickle.dump(lin,open('linreg.pkl','wb'))

model=pickle.load(open('linreg.pkl','rb'))