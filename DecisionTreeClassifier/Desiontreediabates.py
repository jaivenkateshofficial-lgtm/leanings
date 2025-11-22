import pandas as pd
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV

diab=load_diabetes()
print(diab['DESCR'])
df = pd.DataFrame(data=diab.data, columns=diab.feature_names)
x=df
y=pd.DataFrame(diab.target)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=10)
dg=DecisionTreeRegressor()
dg.fit(x_train,y_train)
y_pred=dg.predict(x_test)
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
score=r2_score(y_test,y_pred)


params={'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
        'max_depth':[4,5,6,7,8,9,10]}
gc=GridSearchCV(estimator=dg,param_grid=params,scoring='neg_mean_squared_error')
gc.fit(x_train,y_train)
y_pred_cv=gc.predict(x_test)
mse=mean_squared_error(y_test,y_pred_cv)
mae=mean_absolute_error(y_test,y_pred_cv)
score=r2_score(y_test,y_pred_cv)

# visvalization 
plt.figure(figsize=(15,10))
tree.plot_tree(decision_tree=dg,filled=True)