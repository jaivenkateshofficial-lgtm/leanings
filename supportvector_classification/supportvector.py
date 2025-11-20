import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix

x,y=make_classification(n_samples=1000,n_features=2,n_classes=2,n_clusters_per_class=1,n_informative=2,n_redundant=0,n_repeated=0)
x=pd.DataFrame(x)
y=pd.DataFrame(y)
sns.scatterplot(x=pd.DataFrame(x)[0],y=pd.DataFrame(x)[1],hue=y)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=10)
svc=SVC(kernel='linear')
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
classification=classification_report(y_pred,y_test)
cmatrix=confusion_matrix(y_pred,y_test)
# Assigment use diffrent kernels and get accuracy