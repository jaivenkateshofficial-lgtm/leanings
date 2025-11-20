import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import svm

x=np.linspace(-5.0,5.0,100)
y=np.sqrt(10**2-x**2)
y=np.hstack[y,-y]
x=np.hstack[x,-x]
df1=pd.DataFrame(np.vstack[y,x].T,columns=['X1','X2'])
df1['Y']=0
df2=pd.DataFrame(np.vstack[y,x].T,columns=['X1','X2'])
df2['Y']=0
df=df1.append(df2)
x=df.iloc[:,2]
y=df.Y
