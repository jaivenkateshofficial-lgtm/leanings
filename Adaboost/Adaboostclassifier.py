import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,recall_score,f1_score,roc_auc_score,roc_curve
from sklearn.model_selection import RandomizedSearchCV
df=pd.read_csv(r'data_sets\Travel.csv')
print(df.info())
print(df.isnull().sum())
column_null=[]
for coulm in df.columns:
    if df[coulm].isnull().sum()>0: 
        column_null.append(coulm)
print(column_null)#examining the features
'''
1.Age is continoius value need to replace median(to overcome oulayers)
2.Duration of pich replace with median
3.number of follow ups is float it should be convert to int
4.mothly income median 
'''
for column in column_null:
    if df[column].dtype in ['int64', 'float64']:  # Check if the column is numeric
        df[column].fillna(df[column].median(), inplace=True)
print(df['TypeofContact'].value_counts())
df['TypeofContact'].fillna(df['TypeofContact'].mode()[0],inplace=True)
'''
observation:
1.Age is float can convert to int
2.convering DurationOfPitch to intgers
'''
numerical_columns=[]
catgorical_column=[]
for column in df.columns:
    if df[column].dtype in ['int64', 'float64']:
        numerical_columns.append(column)
    else:
        catgorical_column.append(column)
for column in numerical_columns:
    if (df[column] % 1 == 0).all():
        df[column]=df[column].astype(int)

# Removing of unwanted colums means same columns act as diffrent feature for example number of childervisting and number adult visting
df['Total_peson_visiting']=df['NumberOfPersonVisiting']+df['NumberOfChildrenVisiting']
df.drop(['NumberOfPersonVisiting','NumberOfChildrenVisiting'],inplace=True,axis=1)
df['Gender']=df['Gender'].replace('Fe Male','Female')
df['MaritalStatus']=df['MaritalStatus'].replace('Unmarried','Single')
x=df.drop(['ProdTaken'],axis=1)
y=df['ProdTaken']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=.2,random_state=40)
cat_feature=x.select_dtypes(include='object').columns
num_features=x.select_dtypes(exclude='object').columns


numeric_trasformer=StandardScaler()
one_hot_encoder=OneHotEncoder(drop='first')
ct=ColumnTransformer(transformers=[('one-hotencoder',one_hot_encoder,cat_feature),
                      ('Standard-scalar',numeric_trasformer,num_features)])#This is used comaine multiple transformation and standard scalar is not compesary in random forest
x_train=ct.fit_transform(x_train)
x_test=ct.transform(x_test)

models={
    'logisticregression':LogisticRegression(),
    'desiontree':DecisionTreeClassifier(),
    'knerestneibhour':KNeighborsClassifier(),
    'Randomforest':RandomForestClassifier(),
    'adaboost':AdaBoostClassifier()
}
for i in range(len(list(models))):
    model=list(models.values())[i]
    model.fit(x_train,y_train)
    y_train_predict=model.predict(x_train)
    y_test_predict=model.predict(x_test)

    train_presion=precision_score(y_train,y_train_predict)
    train_accuracy=accuracy_score(y_train,y_train_predict)
    train_cm=confusion_matrix(y_train,y_train_predict)
    train_recall=recall_score(y_train,y_train_predict)
    train_f1=f1_score(y_train,y_train_predict)
    train_roc=roc_auc_score(y_train,y_train_predict)
    print(list(models)[i])
    print(f'train_presion:{train_presion},train_accuracy{train_accuracy},train_cm{train_cm},train_recall{train_recall},train_f1{train_f1},train_roc{train_roc}')

    test_presion=precision_score(y_test,y_test_predict)
    test_accuracy=accuracy_score(y_test,y_test_predict)
    test_cm=confusion_matrix(y_test,y_test_predict)
    test_recall=recall_score(y_test,y_test_predict)
    test_f1=f1_score(y_test,y_test_predict)
    test_roc=roc_auc_score(y_test,y_test_predict)
    print(f'test_presion:{test_presion},test_accuracy{test_accuracy},test_cm{test_cm},test_recall{test_recall},test_f1{test_f1},test_roc{test_roc}')

model=AdaBoostClassifier()
param={
    'n_estimators':[50,60,70,80]
}
rcv=RandomizedSearchCV(estimator=model,param_distributions=param,cv=5,scoring='accuracy')

rcv.fit(x_train,y_train)
y_train_predict=rcv.predict(x_train)
y_test_predict=rcv.predict(x_test)

train_presion=precision_score(y_train,y_train_predict)
train_accuracy=accuracy_score(y_train,y_train_predict)
train_cm=confusion_matrix(y_train,y_train_predict)
train_recall=recall_score(y_train,y_train_predict)
train_f1=f1_score(y_train,y_train_predict)
train_roc=roc_auc_score(y_train,y_train_predict)
print(f'train_presion:{train_presion},train_accuracy{train_accuracy},train_cm{train_cm},train_recall{train_recall},train_f1{train_f1},train_roc{train_roc}')


test_presion=precision_score(y_test,y_test_predict)
test_accuracy=accuracy_score(y_test,y_test_predict)
test_cm=confusion_matrix(y_test,y_test_predict)
test_recall=recall_score(y_test,y_test_predict)
test_f1=f1_score(y_test,y_test_predict)
test_roc=roc_auc_score(y_test,y_test_predict)
print(f'test_presion:{test_presion},test_accuracy{test_accuracy},test_cm{test_cm},test_recall{test_recall},test_f1{test_f1},test_roc{test_roc}')