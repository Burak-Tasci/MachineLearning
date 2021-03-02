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

from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
scaled_x = sc1.fit_transform(X)
sc2 = StandardScaler()
scaled_y = sc2.fit_transform(Y)

from sklearn.svm import SVR

svr = SVR(kernel = 'rbf')
svr.fit(scaled_x,scaled_y)

from sklearn.preprocessing import  PolynomialFeatures
from sklearn.linear_model import LinearRegression

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,Y)

lin_reg = LinearRegression()
lin_reg.fit(X,Y)



from sklearn.metrics import r2_score


print("Random Forest")
r2_value = r2_score(Y,rf_reg.predict(X))
print(r2_value)

print("Decision Tree")
r2_value = r2_score(Y,r_dt.predict(X))
print(r2_value)

print("SVR")
r2_value = r2_score(scaled_y,svr.predict(scaled_x))
print(r2_value)

print("Polynominal")
r2_value = r2_score(Y,lin_reg2.predict(x_poly))
print(r2_value)

print("Linear")
r2_value = r2_score(Y,lin_reg.predict(X))
print(r2_value)












