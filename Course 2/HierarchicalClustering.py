import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

veriler = pd.read_csv('musteriler.csv')

X = veriler.iloc[:,3:].values

from sklearn.cluster import AgglomerativeClustering

ac = AgglomerativeClustering(n_clusters = 3, affinity= 'euclidean', linkage = 'ward')

y_pred = ac.fit_predict(X)

plt.scatter(X[y_pred == 0,0],X[y_pred == 0,1], c = 'red')

plt.scatter(X[y_pred == 1,0],X[y_pred == 1,1], c = 'green')

plt.scatter(X[y_pred == 2,0],X[y_pred == 2,1], c = 'blue')

import scipy.cluster.hierarchy as sch

plt.figure(figsize=(10,5))
dendrogram = sch.dendrogram(sch.linkage(X,method='ward'))
plt.show()

