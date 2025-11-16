import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,Ridge,Lasso,ElasticNet,LassoCV
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

# Feature SelectionS
df=pd.read_csv(r"C:\Users\jaivenkateg\Desktop\practice\data_sets\Algerian_forest_fires_dataset_Cleaned_jai.csv")
df.drop(['day', 'month', 'year'],axis=1,inplace=True)
df['Classes'] = np.where(df['Classes'].str.contains('not fire', na=False), 0, 1)
df['Classes'].value_counts()
x=df.drop(['FWI'],axis=1)
x.head()
y=df['FWI']
# Train test split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
plt.figure(figsize=(12,10))
corr=x_train.corr()
sns.heatmap(corr,annot=True)
plt.show()

# tO FIND Elements which are higly corelated and drop them because two features are higly correlated beteen each other there no use keeping as this feature increase and that feature increse this also increse
def correlation(dataset,threshold):
    col_corr=set()
    corr_matrix=dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i,j])>threshold:
                colname=corr_matrix.columns[i]
                col_corr.add(colname)
    return col_corr
corr=correlation(x_train,0.85)

x_train.drop(corr,inplace=True,axis=1)
x_test.drop(corr,inplace=True,axis=1)
Scalar=StandardScaler()
x_train_scaled=Scalar.fit_transform(x_train)
x_test_scalled=Scalar.transform(x_test)
plt.subplots(figsize=(15,5))
plt.subplot(1,2,1)
sns.boxplot(data=x_train)
plt.title('X_train before Scalling')
plt.subplot(1,2,2)
sns.boxenplot(data=x_test_scalled)
plt.title("X_train After Scalling")
lin=LinearRegression()
lin.fit(x_train_scaled,y_train)
y_pred=lin.predict(x_test_scalled)
mean_squared_error(y_test,y_pred)
r2_score(y_test,y_pred)
# Rige regression 
rig=Ridge()
rig.fit(x_train_scaled,y_train)
y_pred=lin.predict(x_test_scalled)
mean_squared_error(y_test,y_pred)
r2_score(y_test,y_pred)

# lasso
lasso=Lasso()
lasso.fit(x_train_scaled,y_train)
y_pred=lin.predict(x_test_scalled)
mean_squared_error(y_test,y_pred)
r2_score(y_test,y_pred)
# Elastic net
Elstic=ElasticNet()
Elstic.fit(x_train_scaled,y_train)
y_pred=lin.predict(x_test)
mean_squared_error(y_test,y_pred)
r2_score(y_test,y_pred)

# Lasso cv
lcv=LassoCV(cv=5)
lcv.fit(x_test_scalled,y_train)