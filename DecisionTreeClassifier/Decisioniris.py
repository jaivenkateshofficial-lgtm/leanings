import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
'''
Here’s the Iris dataset description reformatted into a clean, structured format:

***

## **Iris Plants Dataset**

### **Data Set Characteristics**

*   **Number of Instances:** 150 (50 in each of three classes)
*   **Number of Attributes:** 4 numeric, predictive attributes + 1 class label
*   **Attribute Information:**
    *   Sepal length (cm)
    *   Sepal width (cm)
    *   Petal length (cm)
    *   Petal width (cm)
    *   Class:
        *   *Iris-Setosa*
        *   *Iris-Versicolour*
        *   *Iris-Virginica*

***

### **Summary Statistics**

| Attribute    | Min | Max | Mean | SD   | Class Correlation |
| ------------ | --- | --- | ---- | ---- | ----------------- |
| Sepal length | 4.3 | 7.9 | 5.84 | 0.83 | 0.7826            |
| Sepal width  | 2.0 | 4.4 | 3.05 | 0.43 | -0.4194           |
| Petal length | 1.0 | 6.9 | 3.76 | 1.76 | 0.9490 (high!)    |
| Petal width  | 0.1 | 2.5 | 1.20 | 0.76 | 0.9565 (high!)    |

***

### **Additional Information**

*   **Missing Attribute Values:** None
*   **Class Distribution:** 33.3% for each of the 3 classes
*   **Creator:** R.A. Fisher
*   **Donor:** Michael Marshall (MARSHALL%<PLU@io.arc.nasa.gov>)
*   **Date:** July, 1988

***

### **Notes**

*   The dataset was first used by Sir R.A. Fisher in his classic paper on pattern recognition.
*   One class is linearly separable from the other two; the latter two are **not** linearly separable from each other.
*   This dataset is widely used in machine learning and statistics education.

***

### **References**

*   Fisher, R.A. *The use of multiple measurements in taxonomic problems*. Annual Eugenics, 7, Part II, 179-188 (1936).
*   Duda, R.O., & Hart, P.E. (1973). *Pattern Classification and Scene Analysis*. John Wiley & Sons.
*   Dasarathy, B.V. (1980). *Nosing Around the Neighborhood: A New System Structure and Classification Rule for Recognition in Partially Exposed Environments*. IEEE Transactions on Pattern Analysis and Machine Intelligence.
*   Gates, G.W. (1972). *The Reduced Nearest Neighbor Rule*. IEEE Transactions on Information Theory.
*   Cheeseman et al. (1988). *AUTOCLASS II conceptual clustering system finds 3 classes in the data*.

***

'''

iris=load_iris()
iris['DESCR']
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df.columns = df.columns.str.replace(' \(cm\)', '', regex=True)
df.columns.str.strip(' ')
x=df[['sepal length','sepal width','petal length','petal width']]
df['target'] = iris.target
y=df['target']
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=10)
dc=DecisionTreeClassifier()#This doesn't need any scalling algorithums
dc.fit(x_train,y_train)
y_pred=dc.predict(x_test)
cl=classification_report(y_test,y_pred)
ac=accuracy_score(y_pred,y_test)
cm=confusion_matrix(y_test,y_pred)

# Visvalizing the desion tree and post prooning 
plt.figure(figsize=(15,10))
tree.plot_tree(dc,filled=True)
plt.show()
# From this we came to know max_depth =2 is enfough we can do that and prone tree
dc=DecisionTreeClassifier(max_depth=2)#This doesn't need any scalling algorithums
dc.fit(x_train,y_train)
y_pred=dc.predict(x_test)
cl=classification_report(y_test,y_pred)
ac=accuracy_score(y_pred,y_test)
cm=confusion_matrix(y_test,y_pred)

# post prooning
dc=DecisionTreeClassifier()
params={'criterion':['gini', 'entropy', 'log_loss'],
        'max_depth':[1,2,3,4,5],'min_samples_split':[1.1,1.2,2],'min_impurity_decrease':[0,1,2,3]}
gc=GridSearchCV(estimator=dc,scoring='accuracy',param_grid=params,cv=5)
gc.fit(x_train,y_train)
print(f'The best paran{gc.best_params_}.the best score:{gc.best_score_}')

y_pred=gc.predict(x_test)
cl=classification_report(y_test,y_pred)
ac=accuracy_score(y_pred,y_test)
cm=confusion_matrix(y_test,y_pred)

