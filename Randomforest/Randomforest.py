'''
Handling missing values
Handling duplicates
Check Data
understanding the dataset
'''
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,precision_score,confusion_matrix,recall_score,f1_score,roc_auc_score,roc_curve
df=pd.read_csv(r'data_sets\Travel.csv')
print(df.info())
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
if (df['DurationOfPitch'] % 1 == 0).all():
    print("All values in the column are integers (though stored as floats).")
df['DurationOfPitch']=df['DurationOfPitch'].astype(int)
df['Age']=df['Age'].astype(int)
df['NumberOfFollowups']=df['NumberOfFollowups'].astype(int)
df['NumberOfChildrenVisiting']=df['NumberOfChildrenVisiting'].astype(int)
if (df['PreferredPropertyStar'] % 1 == 0).all():
    print("All values in the column are integers (though stored as floats).")
df['PreferredPropertyStar']=df['PreferredPropertyStar'].astype(int)
df['MonthlyIncome']=df['MonthlyIncome'].astype(int)

# Removing of unwanted colums means same columns act as diffrent feature for example number of childervisting and number adult visting
df['Total_peson_visiting']=df['NumberOfPersonVisiting']+df['NumberOfChildrenVisiting']
df.drop(['NumberOfPersonVisiting','NumberOfChildrenVisiting'],inplace=True,axis=1)
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
# Random forest classifier
