import pandas as pd
import numpy as np
from sklearn.utils import resample 

toatal_data=1000
n_class_a= int(toatal_data*.90)
n_class_b=toatal_data-n_class_a
class_a=pd.DataFrame({
    'feature1':np.random.normal(loc=0,scale=1,size=n_class_a),
    'feature2':np.random.normal(loc=0,scale=1,size=n_class_a),
    'target':[0]*n_class_a
})

class_b=pd.DataFrame({
    'feature1':np.random.normal(loc=0,scale=1,size=n_class_b),
    'feature2':np.random.normal(loc=0,scale=1,size=n_class_b),
    'target':[1]*n_class_b
})
print(class_b['target'].value_counts())
df=pd.concat([class_a,class_b])
df_minory=df[df['target']==1]
df_majority=df[df['target']==0]
print(df_majority)
print(df_minory)
df_minory=resample(df_minory,replace=True,n_samples=n_class_a)
print(df_minory.shape)