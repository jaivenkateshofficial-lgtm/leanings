import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

x=[1,2,3,4,5,6,7,8,9,10]
y=[1,4,9,16,25,36,49,64,81,100]
plt.plot(x,y)#basic line plot
plt.xlabel('naturan numbers')
plt.ylabel('square numbers')
plt.title('Square number realation ship')

# customised line plot
plt.plot(x,y,c="yellow",linestyle="--",marker='o',linewidth=2,markersize=9)
plt.grid(True)

# Muti plots
x=[1,2,3,4,5,6,7,8,9,10]
y1=[1,4,9,16,25,36,49,64,81,100]
y2=[2,4,6,8,10,12,14,16,18,20]
plt.figure(figsize=(9,5))
plt.subplot(2,2,1)#one row and 2 column in total figare we are divinding into 2 row and 2 colum #which plot we need to put here
plt.plot(x,y1,c='green')
plt.subplot(2,2,2)
plt.xlabel("natural nubers")
plt.ylabel("multiple of 2")
plt.title("liniar model prediction")
plt.plot(x,y2,c='orange')
plt.subplot(2,2,3)#third plot
plt.xlabel("natural nubers")
plt.ylabel("multiple of 2")
plt.title("liniar model prediction")
plt.plot(x,y2,c='orange')

# Bar plot
plt.figure(figsize=(9,5))
catagories=['A','B','C','D','E','F']
values=[10,11,12,13,14,25]
plt.bar(catagories,values,color="blue")

# Histogram
plt.figure(figsize=(9,5))
data=[1,1,1,2,2,3,3,3,3,3,4,5,6,7,7,7,8,8,9,9,9,9,10]
plt.hist(data,bins=10,color='yellow',edgecolor='black')
ax=plt.gca()
ax.set_aspect(aspect=2, adjustable='box')
plt.show()
# creating scatter plot
plt.figure(figsize=(9,5))
plt.scatter(x,y)

# pie chart
plt.figure(figsize=(9,5))
labels=['A','B','C','D']
sizes=[30,40,20,10]
colour=['Red','green','Blue','yellow']
explode=(0.2,0,0,0)#move out the first slice
plt.pie(sizes,labels=labels,colors=colour,explode=explode,autopct='%1.1f%%',shadow=True)

# Seaborn is visvalization library which build in top of matplot lib for advanced visvalization
tips=sns.load_dataset('tips')
sns.scatterplot(x='total_bill',y='tip',data=tips)
plt.title('Scatter plot of toatal bill vs tips')
plt.show()
# Catogorical varibale
sns.barplot(x='day',y='tip',data=tips)
plt.show()
# Box plot
# This used to determaine the outlayers
sns.boxplot(x='day',y='tip',data=tips)
plt.show()

# Voilen plot
# It is used detemine the distribution of the data
sns.violinplot(x='day',y='tip',data=tips)
plt.show()

# Historgarm
sns.histplot(tips['total_bill'],bins=10,kde=True)
plt.show()

# kde plot
sns.kdeplot(tips['total_bill'],shade=True,color='yellow')
plt.show()

# pair plot
# each to get reatationship between each every column
sns.pairplot(tips)
plt.show()