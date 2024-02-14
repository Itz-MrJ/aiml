
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv('ch1ex1.csv')
points = df.values
print(points)

from sklearn.cluster import KMeans
model = KMeans(n_clusters=3)
model.fit(points)
labels = model.predict(points)

xs = points[: , 0]
ys = points[:,1]
print(xs)
print(ys)

plt.scatter(xs, ys,c=labels)
plt.show()

centroids = model.cluster_centers_

centroids_x = centroids[:,0]
centroids_y = centroids[:,1]

plt.scatter(xs, ys, c=labels)
plt.scatter(centroids_x, centroids_y, marker='*', s=200)
plt.show()
