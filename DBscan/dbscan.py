from sklearn.cluster import DBSCAN
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

x,y=make_moons(n_samples=250,noise=0.05)
scalar=StandardScaler()
x_scaled=scalar.fit_transform(x)
db=DBSCAN(eps=0.3)
db.labels_
# plot with respect to label.
