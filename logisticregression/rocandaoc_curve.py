from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,roc_curve
import  matplotlib.pyplot as plt
import numpy as np

x,y =make_classification(n_samples=1000,n_classes=2,random_state=1)
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,test_size=0.25)

# created dumy model with 0 has output
dumy_model=[0 for _ in range(len(y_test))]

model=LogisticRegression()
model.fit(x_train,y_train)

# Prediction interms of probality
model_prob=model.predict_proba(x_test)
model_prob=model_prob[:,1]#The numpy array for 2d can be acessed by a[0,1]
dummy_auc=roc_auc_score(y_test,dumy_model)
model_auc=roc_auc_score(y_test,model_prob)
print(f'dummy_auc:{dummy_auc},model_auc{model_auc}')
dummy_fpr,dummy_tpr,threshold=roc_curve(y_test,dumy_model)
model_fpr,model_tpr,threshold=roc_curve(y_test,model_prob)
fig=plt.figure(figsize=(20,50))
plt.plot(dummy_fpr,dummy_tpr,linestyle='--',label='Dummy model')
plt.plot(model_fpr,model_tpr,linestyle='--',label='Logistic model')
plt.xlabel("False postive")
plt.ylabel("True positive")
ax=fig.add_subplot(111)
for xyz in zip(model_fpr,model_tpr,threshold):
    ax.annotate("%s"%np.round(xyz[2],2),xy=(xyz[0],xyz[1]))
plt.show()
# This cureve will give the probality of tp and fp for diffrent threshold values we select the point were high tp and less fp