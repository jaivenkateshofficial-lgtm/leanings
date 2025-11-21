import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error,mean_squared_error


df=sns.load_dataset('tips')
print(df.columns)
df['smoker'].value_counts()
df['sex'].value_counts()
df['day'].value_counts()
df['time'].value_counts()
df['size'].value_counts()

lc1=LabelEncoder()
lc2=LabelEncoder()
lc3=LabelEncoder()
x=df[['day','smoker','tip','size','time','sex']]
y=df[['total_bill']]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=10)
x_train['smoker']=lc1.fit_transform(x_train['smoker'])
x_train['sex']=lc2.fit_transform(x_train['sex'])
x_train['time']=lc3.fit_transform(x_train['time'])
encoder=OneHotEncoder()
ct=ColumnTransformer(transformers=[('one-hotencoding',OneHotEncoder(),[0])],remainder='passthrough')
x_train=ct.fit_transform(x_train)

x_test['smoker']=lc1.transform(x_test['smoker'])
x_test['sex']=lc2.transform(x_test['sex'])
x_test['time']=lc3.transform(x_test['time'])
encoder=OneHotEncoder()
ct=ColumnTransformer(transformers=[('one-hotencoding',OneHotEncoder(),[0])],remainder='passthrough')
x_test=ct.fit_transform(x_test)

rg=Ridge()
rg.fit(x_train,y_train)
y_pred=rg.predict(x_test)
mse=mean_squared_error(y_test,y_pred)




