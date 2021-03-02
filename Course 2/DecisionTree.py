
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('maaslar.csv',sep=";")
egitim = veriler.iloc[:,1:2]
maas = veriler.iloc[:,2:]

X = egitim.values
Y = maas.values

from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

plt.scatter(X,Y,color="red")
plt.plot(X,r_dt.predict(X),color="blue") 


Z = X + 0.5
K = X - 0.4


print(r_dt.predict([[11]]))
print(r_dt.predict([[6.6]]))
plt.plot(egitim,r_dt.predict(Z),color="green")
plt.plot(egitim,r_dt.predict(K),color = "yellow")
