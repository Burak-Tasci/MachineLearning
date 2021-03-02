import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('maaslar.csv',sep=";")

egitim = veriler.iloc[:,1:2]
maas = veriler.iloc[:,2:]

X = egitim.values
Y = maas.values

from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
scaled_x = sc1.fit_transform(X)
sc2 = StandardScaler()
scaled_y = sc2.fit_transform(Y)

from sklearn.svm import SVR

svr = SVR(kernel = 'rbf')
svr.fit(scaled_x,scaled_y)

plt.scatter(scaled_x, scaled_y,color="red")
plt.plot(scaled_x,svr.predict(scaled_x),color="blue")



