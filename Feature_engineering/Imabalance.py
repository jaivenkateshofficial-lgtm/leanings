from sklearn.datasets import make_classification
import pandas as pd
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

x,y=make_classification(n_samples=1000,n_clusters_per_class=1,weights=[.90],n_redundant=0,random_state=12,n_features=2)
df1=pd.DataFrame(x,columns=['f1','f2'])
df2=pd.DataFrame(y,columns=['target'])
final_df=pd.concat([df1,df2],axis=1)
print(final_df['target'].value_counts())
print(final_df.head())
plt.scatter(final_df['f1'],final_df['f2'],c=final_df['target'])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Imbalanced Classification Data')
plt.show()
oversample=SMOTE()
x,y=oversample.fit_resample(final_df[['f1','f2'],final_df['target']])
plt.scatter(final_df['f1'],final_df['f2'],c=final_df['target'])
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Imbalanced Classification Data')
plt.show()
