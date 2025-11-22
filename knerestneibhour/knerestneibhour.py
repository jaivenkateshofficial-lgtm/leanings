import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix

kd=KNeighborsClassifier(n_neighbors=6,algorithm='kd_tree',p=2)#p=2 eqalinear distance by default itself true

x,y=make_classification(n_samples=1000,n_classes=2,n_features=10,random_state=10)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=10)
kd.fit(x_train,y_train)
y_pred=kd.predict(x_test)
cl=classification_report(y_test,y_pred)
ac=accuracy_score(y_pred,y_test)
cm=confusion_matrix(y_test,y_pred)
param={'n_neighbors':[1,2,3,4,5,6,7,8.9,10],'p':[1,2],'algorithm':['auto', 'ball_tree','kd_tree','brute']}
gc=GridSearchCV(estimator=kd,param_grid=param)
gc.fit(x_train,y_train)
gc.best_params_
y_predgc=gc.predict(x_train)
cl_gc=classification_report(y_test,y_pred)
ac_gc=accuracy_score(y_pred,y_test)
cm_gc=confusion_matrix(y_test,y_pred)