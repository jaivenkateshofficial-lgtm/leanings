import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler

x,y=make_blobs(n_samples=1000,centers=3,n_features=2)
plt.scatter(x[:,0],x[:,1],c=y)
