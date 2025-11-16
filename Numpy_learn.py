import numpy as np

arr1=np.array([1,2,3,4,5])
print(arr1)
print(type(arr1))
print(arr1.shape)
arr2=np.array([1,2,3,4,5])
arr2=arr2.reshape(1,5)
print(arr2)
print(arr2.shape)
a=np.arange(0,10,2).reshape(5,1)
print(a)
one_arr=np.ones((3,4))
print(arr2.ndim)#To print the number of dimention of an array
print(arr2.dtype)#print the data type
print(arr2.size)
print(one_arr)
# Numpy vecorized addition
arr1=np.array([1,2,3,4,5])
arr2=np.array([1,2,3,4,5])
print(arr1+arr2)
l1=[1,2,3,4,5]
l2=[1,2,3,4,5]
print(l1+l2)#ot will act as concarination in list but it act as an addition of all elements in arr due to ints same data type
print(arr1-arr2)
print(arr1*arr2)
print(arr1/arr2)
print(arr1%arr2)
print(arr1**arr2)
# universal function
print(np.sqrt(arr1))
print(np.exp(arr1))
print(np.sin(arr1))
print(np.log(arr1))
# Acessing elements in array
arr2=np.array([[1,2,3,4],[5,6,7,8]])
print(arr2[0])
print(arr2[0][0])
print(arr2[0:])
print(arr2[0:,1:])
# Modify element in arr
arr2[0][0]=100
print(arr2)
# There are two methods of accesing multidimentional array arr[0][0] or arr[0,0]
arr2[1:]=50#chnage all the elements
print(arr2)
# Statistical Normalization
# Normalization
data=np.array([11,12,13,14])
mean=np.mean(data)
std_dev=np.std(data)
normalization=(data-mean)/std_dev
print(normalization)
print(data>5)
print(data[data>5])
print(data[(data>2) & (data<5)])