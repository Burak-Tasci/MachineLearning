
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


veriler = pd.read_csv('veriler.csv')



print(veriler)

BoyKiloYas = veriler.iloc[:,1:4].values

ulke = veriler.iloc[:,0:1].values

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])
ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()

cinsiyet = veriler.iloc[:,-1:].values
cinsiyet = ohe.fit_transform(cinsiyet).toarray()
cinsiyet = pd.DataFrame(cinsiyet)
cinsiyetK = np.array(cinsiyet.drop(1,axis = 1))

ulke = pd.DataFrame(data = ulke, index = range(22), columns = ['fr','tr','us'])

cinsiyetE = pd.DataFrame(data = cinsiyetK,index = range(22),columns = ['cinsiyet'])

BoyKiloYas = pd.DataFrame(data = BoyKiloYas, index = range(22),columns = ['boy','kilo','yas'])

Inputs = pd.concat([ulke,BoyKiloYas],axis = 1)

Toplam = pd.concat([Inputs,cinsiyetE],axis = 1)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(Inputs,cinsiyetE, random_state =0, test_size = 0.33)

from sklearn.linear_model import  LinearRegression

regresor = LinearRegression()
regresor.fit(x_train,y_train)

y_pred = regresor.predict(x_test)

boy = Toplam.iloc[:,3:4].values
sol = Toplam.iloc[:,:3]
sag = Toplam.iloc[:,4:]

solvesag = pd.concat([sol,sag],axis = 1)

x_train, x_test, y_train, y_test = train_test_split(solvesag,boy, random_state =0, test_size = 0.33)

Regressor2 = LinearRegression()
Regressor2.fit(x_train,y_train)

y_pred = Regressor2.predict(x_test)

import statsmodels.api as sm

X = np.append(np.ones((22,1)).astype(int),values =  solvesag,axis = 1)

XList = solvesag.iloc[:,:].values
Xlist = np.array(XList,dtype = float)
model = sm.OLS(boy,XList).fit()

XList = solvesag.iloc[:,[0,1,2,3,5]].values
Xlist = np.array(XList,dtype = float)
model = sm.OLS(boy,XList).fit()

XList = solvesag.iloc[:,[0,1,2,3]].values
XList = np.array(XList,dtype = float)
model = sm.OLS(boy,XList).fit()

print(model.summary())


x_train, x_test, y_train, y_test = train_test_split(XList,boy, random_state =0, test_size = 0.33)



Regressor3 = LinearRegression()
Regressor3.fit(x_train,y_train)

y_pred = Regressor3.predict(x_test)






























