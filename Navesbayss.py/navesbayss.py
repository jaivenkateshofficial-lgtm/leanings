import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix


x,y=load_iris(return_X_y=True)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=10)

gb=GaussianNB()
gb.fit(x_train,y_train)
y_pred=gb.predict(x_test)

ca=classification_report(y_test,y_pred)
ac=accuracy_score(y_test,y_pred)
cm=confusion_matrix(y_test,y_pred)
