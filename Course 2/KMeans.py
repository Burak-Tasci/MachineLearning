
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D


veriler = pd.read_csv('musteriler.csv')

#2 Boyutlu
X = veriler.iloc[:,3:].values
sonuclar = []

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 3, init = 'random')
y_pred = kmeans.fit_predict(X)
print(kmeans.cluster_centers_)
print()
print()
noktalar = kmeans.cluster_centers_

for i in range(1,10):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 0)
    kmeans.fit(X)
    sonuclar.append(kmeans.inertia_)
    # print(kmeans.cluster_centers_)

plt.figure(0)
plt.plot(range(1,10),sonuclar)
plt.show()


print()
print()
print(sonuclar)

data_x = pd.DataFrame(data = X,index = range(len(X)),columns = ['yas','hacim'])

noktalar = pd.DataFrame(data = noktalar,index = range(len(noktalar)),columns = ['x','y'])

print()
plt.figure(1)


category_0 = y_pred == 0
category_1 = y_pred == 1
category_2 = y_pred == 2

sns.scatterplot(data = data_x[category_0], x ='yas',y ='hacim', color = 'green')
sns.scatterplot(data = data_x[category_1], x ='yas',y ='hacim', color = 'blue')
sns.scatterplot(data = data_x[category_2], x ='yas',y ='hacim', color = 'red')

sns.scatterplot(data = noktalar,x = 'x', y = 'y', color = 'black')
plt.show()




