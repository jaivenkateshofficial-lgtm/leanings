import seaborn as sns
import matplotlib.pyplot as plt 
import pandas as pd

# arr= [55,65,75,85,55,45,65,75,85,55,95,45,35,25]
# sns.boxenplot(arr)
# plt.show()
data={
    'name':["jai",'sai','Roshan','sujan','Bharath'],
    'age':[24,22,18,11,26],
    'Gender':['male','male','male','male','male'],
    'mothertounge':['Tamil','Tamil','Tamil','Tamil','Tamil']
}
df1=pd.DataFrame(data)
print(df1[['age','name']])
