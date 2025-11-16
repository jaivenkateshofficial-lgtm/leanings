import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
x=6*np.random.rand(100,1)-3
y=0.5*x**2+1.5*x+2+np.random.rand(100,1)
plt.scatter(x,y,c='g')
plt.xlabel('xdataset')
plt.ylabel('ydataset')
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.2)
reg=LinearRegression()
reg.fit(x_train,y_train)
score= r2_score(y_test,reg.predict(x_test))
print(score)
plt.plot(x_train,reg.predict(x_train),c='r')
plt.scatter(x_train,y_train)
plt.xlabel('xdataset')
plt.ylabel('ydataset')

# Polynomial regression
poly=PolynomialFeatures(degree=2,include_bias=True)#it will include bias
x_train_poly=poly.fit_transform(x_train)
x_test_poly=poly.transform(x_test)
reg.fit(x_train_poly,y_train)
score2= r2_score(y_test,reg.predict(x_test_poly))
print(r2_score)
# prediction
x_new=np.linspace(-3,3,200).reshape(200,1)
x_new_poly=poly.transform(x_new)

# Combaining polynomial feature with liniar regression is know as piplining
def poly_regression(degree):
    x_new=np.linspace(-3,3,200).re(200,1)
    poly_feature=PolynomialFeatures()
    lin_reg=LinearRegression()
    poly_reg=poly_regression=Pipeline([("poly_feature",poly_feature),("lin_reg",lin_reg)])
    poly_reg.fit(x_train,y_train)
    y_pred=poly_reg.predict(x_new)
    plt.plot(x_new,y_pred)
    plt.plot(x_train,y_train)
    plt.plot(x_test,y_test)
    plt.legend(loc="upper left")
    plt.xlabel('x data')
    plt.xlabel('y data')
    plt.axis(-4,-4,0,10)
    plt.show()

poly_regression(5)
print('hello')