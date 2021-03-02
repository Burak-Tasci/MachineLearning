

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('maaslar_yeni.csv',sep=",")

X = veriler.iloc[:,1:-1]
Y = veriler.iloc[:,-1:].values

Unvan = veriler.iloc[:,1:2].values
print(Unvan)

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

Unvan[:,0] = le.fit_transform(veriler.iloc[:,2].values)
ohe = preprocessing.OneHotEncoder()
Unvan = ohe.fit_transform(Unvan).toarray()
print(Unvan)

Unvan = pd.DataFrame(data = Unvan, index = range(30),columns= ['Cayci','Sekreter','Uzman Yardimcisi', 'Uzman', 'Proje Yoneticisi', 'Sef', 'Mudur', 'Direkter', 'C-level','CEO',])
Diger = pd.DataFrame(data = X.iloc[:,2:].values,index = range(30), columns = ['Kidem','Puan'])

print(Unvan)

Inputs = pd.concat([Unvan,Diger], axis = 1)
Outputs  = pd.DataFrame(data = Y, index = range(30), columns = ['Maas'])

SumOf = pd.concat([Inputs,Outputs], axis = 1)

prediction = [[0,0,0,0,0,0,0,0,0,1,10,100]]

#Multiple Linear Regression
from sklearn.linear_model import  LinearRegression

mlr_regressor = LinearRegression()
mlr_regressor.fit(Inputs, Outputs)
mlr_pred = mlr_regressor.predict(prediction)

import statsmodels.api as sm
model = sm.OLS(mlr_regressor.predict(Inputs),Inputs)
print(model.fit().summary())


#Polynomial Regression

from sklearn.preprocessing import  PolynomialFeatures

poly_reg = PolynomialFeatures(degree=4)
x_poly = poly_reg.fit_transform(Inputs)

lin_reg2 = LinearRegression()
lin_reg2.fit(x_poly,Outputs)

poly_pred = lin_reg2.predict(poly_reg.fit_transform(prediction))

model2 = sm.OLS(lin_reg2.predict(poly_reg.fit_transform(Inputs)),Inputs)
print(model2.fit().summary())

#Support Vector Regression


from sklearn.preprocessing import StandardScaler

sc1 = StandardScaler()
scaled_x = sc1.fit_transform(Inputs)
sc2 = StandardScaler()
scaled_y = sc2.fit_transform(Outputs)

#SVR
from sklearn.svm import SVR

svr = SVR(kernel = 'rbf')
svr.fit(scaled_x,scaled_y)
svr_pred = svr.predict(sc1.fit_transform(prediction))

svr_pred = sc2.inverse_transform(svr_pred)

#Decision Tree
from sklearn.tree import DecisionTreeRegressor

r_dt = DecisionTreeRegressor(random_state=0)
r_dt.fit(Inputs,Outputs)

dt_pred = r_dt.predict(prediction)

#Random Forest
from sklearn.ensemble import RandomForestRegressor

rf_reg = RandomForestRegressor(random_state=0,n_estimators=10)
rf_reg.fit(Inputs,Outputs)

rf_pred = rf_reg.predict(prediction)




















