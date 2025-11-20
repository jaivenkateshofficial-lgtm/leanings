import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.model_selection import GridSearchCV,StratifiedGroupKFold,RandomizedSearchCV

x,y=make_classification(n_samples=1000,n_features=10,n_classes=2,random_state=42)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=42)
logistic=LogisticRegression()
logistic.fit(x_train,y_train)
y_pred=logistic.predict(x_test)

score=accuracy_score(y_test,y_pred)
print(score)
cm=confusion_matrix(y_test,y_pred)
print(cm)
# Accurancy=TP+FP/(TP+FP+TN+FN)
# Precision=TP/(TP+FP)
'''
Hyperparameter tuning and cross validation
'''
# GridSearchCV
model=LogisticRegression()
penalty=['l1','l2','elasticnet']
c_values=[100,10,1.0,0.1,0.01]
solver=['newton-cg','lbfgs','liblinear','sag','saga']
para=dict(penalty=penalty,C=c_values,solver=solver)
cv=StratifiedGroupKFold()
grid=GridSearchCV(estimator=model,param_grid=para,scoring='accurancy',cv=cv,n_jobs=-1)#n_jobs -it will use all proccessor in the system
grid.fit(x_train,y_train)
print(grid.best_params_)
print(grid.best_score_)
y_pred=grid.predict(x_test)
score=accuracy_score(y_test,y_pred)
print(score)
cm=confusion_matrix(y_test,y_pred)
print(cm)
# Hyper parameter tuning used to determaine the best parameter to be used such as alpha ,C,solver
# It will take more time as it is doing for entire sample

# RandomiserCV
randomcv=RandomizedSearchCV()
model=LogisticRegression()
reandomcv=RandomizedSearchCV(estimator=model,param_distributions=para,cv=5,scoring='accuracy')
randomcv.fit(x_test,y_train)
print(randomcv.best_score_)
print(randomcv.best_params_)
y_pred=randomcv.predict(x_test)
score=accuracy_score(y_test,y_pred)
print(score)
cm=confusion_matrix(y_test,y_pred)
print(cm)