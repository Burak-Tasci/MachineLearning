import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('maaslar.csv',sep=";")
egitim = veriler.iloc[:,1:2]
maas = veriler.iloc[:,2:]

X = egitim.values
Y = maas.values

Z = X + 0.5
K = X - 0.4

from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(random_state=0,n_estimators=10)
rf_reg.fit(X,Y.ravel())
print(rf_reg.predict([[6.6]]))

plt.scatter(X,Y,color="red")
plt.plot(X,rf_reg.predict(X),color="blue")
plt.plot(X,rf_reg.predict(Z),color="green")
plt.plot(X,rf_reg.predict(K),color="yellow")













