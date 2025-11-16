import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import statsmodels.api as sm


df=pd.read_csv(r'C:\Users\jaivenkateg\Desktop\practice\data_sets\height-weight.csv')
# plt.scatter(x=df['Weight'],y=df['Height'])
# plt.xlabel("weight")
# plt.ylabel('height')
# plt.show()
# print("hello")
# sns.pairplot(df)
# we need alwas df[['Weight']] to get data frame df['weight'] will give array we can either used 2d array or dataframe
x=df[['Weight']]
y=df['Height']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.25,random_state=42)
# We need to standardization which makes mean =0 and stadad deviation =1 zscore=(xi-mu)/std
scalar=StandardScaler()
x_train=scalar.fit_transform(x_train)
# I will fit_transform for x_train and transform for x_test which means fit_transform calculate the mean for train and standard deviation of train but trasform use train mean and standard deviation for transform
# This done to prevent data lekage
x_test=scalar.transform(x_test)
regression=LinearRegression()
regression.fit(x_train,y_train)
print(regression.coef_)
print(regression.intercept_)
# plot best fit line
plt.scatter(x_train,y_train)
plt.scatter(x_train,regression.predict(x_train))
plt.plot(x_train,regression.predict(x_train))
# Pridicting the values of data
# predicted height will be =incept+coefficent*weight
plt.plot(x_train,regression.predict(x_test))
y_pred=regression.predict(x_test)

# perfomance Matrcs
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
rmse=np.sqrt(mean_squared_error(y_test,y_pred))
print(f'mse:{mse},mae:{mae},rmse:{rmse}')
score=r2_score(y_test,y_pred)
print(f'The r2score is {r2_score}')

# ols liniar regression
model=sm.OLS(y_train,x_train).fit()
model.predict(x_test)
model.summary()
# predicting new weight value
regression.predict([scalar.transform([72])])