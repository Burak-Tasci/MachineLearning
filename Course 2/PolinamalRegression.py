import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('maaslar.csv',sep = ";")

egitim = veriler.iloc[:,1:2]
maas = veriler.iloc[:,2:]

X = egitim.values
Y = maas.values

from sklearn.linear_model import LinearRegression

#linear regression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)
plt.scatter(X,maas,color = 'red')
plt.plot(X,lin_reg.predict(X), color = 'blue')
plt.show()

#polynomial regression
from sklearn.preprocessing import  PolynomialFeatures

poly_reg = PolynomialFeatures(degree=2)
x_poly = poly_reg.fit_transform(X)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,Y)
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(x_poly), color = 'blue')
plt.show()

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(X)
lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,Y)
plt.scatter(X,Y,color = 'red')
plt.plot(X,lin_reg2.predict(x_poly), color = 'blue')
plt.show()

# poly_reg = PolynomialFeatures(degree=8)
# x_poly = poly_reg.fit_transform(X)
# lin_reg2 = LinearRegression()
# lin_reg2.fit(x_poly,Y)
# plt.scatter(X,Y,color = 'red')
# plt.plot(X,lin_reg2.predict(x_poly), color = 'blue')
# plt.show()

#tahminler
print(lin_reg.predict([[11]]))
print(lin_reg.predict([[6.6]]))

print(lin_reg2.predict(poly_reg.fit_transform([[11]])))
print(lin_reg2.predict(poly_reg.fit_transform([[6.6]])))


