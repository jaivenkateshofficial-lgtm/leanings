import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.model_selection import GridSearchCV,StratifiedGroupKFold,RandomizedSearchCV

x,y=make_classification(n_samples=1000,n_features=10,n_classes=2,random_state=42)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.30,random_state=42)
logistic=LogisticRegression(multi_class='ovr')
logistic.fit(x_train,y_train)
y_pred=logistic.predict(x_test)
score=accuracy_score(y_test,y_pred)
print(score)
cm=confusion_matrix(y_test,y_pred)
print(cm)
print(classification_report(y_test,y_pred))

# unbalced data set
x,y=make_classification(n_samples=10000,n_features=10,n_classes=2,n_clusters_per_class=1,random_state=42,weights=[0.99])

penalty=['l1','l2','elasticnet']
c_values=[100,10,1.0,0.1,0.01]
solver=['newton-cg','lbfgs','liblinear','sag','saga']
class_weight=[{0:w ,1:y} for w in [1,10,50,100] for y in [1,10,50,100]]
para=dict(penalty=penalty,C=c_values,solver=solver,class_weight=class_weight)
cv=StratifiedGroupKFold()
grid=GridSearchCV(estimator=LogisticRegression(),param_grid=para,scoring='accuracy',cv=cv,n_jobs=-1)#n_jobs -it will use all proccessor in the system
grid.fit(x_train,y_train)
print(grid.best_params_)
print(grid.best_score_)
y_pred=grid.predict(x_test)
score=accuracy_score(y_test,y_pred)
