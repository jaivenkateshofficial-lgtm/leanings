import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
import seaborn as sns

df =pd.read_csv(r'C:\Users\jaivenkateg\Desktop\practice\data_sets\height-weight.csv')
plt.scatter(df['Weight'],df['Height'])
plt.xlabel('Wegiht')
plt.ylabel("Height")
# spliting feature
x=df[['Weight']]#inpdependet feature
y=df[['Height']]#depandent feature
# Train and test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20,random_state=42)
# Standardized of independent feature
scalar=StandardScaler()
x_train=scalar.fit_transform(x_train)#This comute mean and standard deviation and compute the zcore for all data points
x_test=scalar.transform(x_test)#use the same mean and std of train 

lin=LinearRegression()
lin.fit(x_train,y_train)
print(lin.coef_)
lin.intercept_
plt.scatter(x_train,y_train)
plt.plot(x_train,lin.predict(x_train),c='orange')
y_pred_test=lin.predict(x_test)
plt.scatter(x_test,y_pred_test,c="yellow")
# Perfomance Matrix
# MSE,MAE,RMSE
# R2 and ajucested r2
mse=mean_squared_error(y_test,y_pred_test)
mae=mean_absolute_error(y_test,y_pred_test)
rmse=np.sqrt(mse)
rscore=r2_score(y_test,y_pred_test)
# adjusted r2
# Formila
adjscore=1-(1-rscore)*(len(y_test)-1)/(len(y_test)-x_test.shape[1]-1)
scaled_weight=scalar.transform([[80]])
lin.predict(scaled_weight)
# Asseptions
'''
1.scatterploat plot for prediction: will see linear points
'''
plt.scatter(y_test,y_pred_test)
'''
2.Residuals:will see 
'''
res=y_test-y_pred_test
sns.displot(res,kde=True)

'''
3.scatter plot with respect to predection and residuals
This should be an uniform distribution
'''
plt.scatter(y_pred_test,res)
