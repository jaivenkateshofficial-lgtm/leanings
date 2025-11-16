import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
import pipe_lining


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