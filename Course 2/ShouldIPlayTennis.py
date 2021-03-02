# -*- coding: utf-8 -*-
"""
Created on Mon Jan  4 21:53:14 2021

@author: ABRA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


veriler = pd.read_csv('odev_tenis.csv')

Inputs = veriler.iloc[:,:-1].values
Outputs = veriler.iloc[:,-1:].values

TemperatureAndHumidity = veriler.iloc[:,1:3]

from sklearn import preprocessing

#veriler2 = veriler.apply(LabelEncoder().fit_transform())

le = preprocessing.LabelEncoder()

outlook = veriler.iloc[:,0:1].values
outlook[:,0] = le.fit_transform(veriler.iloc[:,0])
ohe = preprocessing.OneHotEncoder()
outlook = ohe.fit_transform(outlook).toarray()

windy = veriler.iloc[:,-2:-1].values
windy[:,0] = le.fit_transform(veriler.iloc[:,-2])
windy = ohe.fit_transform(windy).toarray()
windy = pd.DataFrame(windy)
windy = np.array(windy.drop(0,axis = 1))

play = veriler.iloc[:,-1:].values
play[:,0] = le.fit_transform(veriler.iloc[:,-1])
play = ohe.fit_transform(play).toarray()
play = pd.DataFrame(play)
play = np.array(play.drop(0,axis = 1))


outlook = pd.DataFrame(data = outlook,index = range(len(outlook)),columns = ['overcast','rainy','sunny'])

windy = pd.DataFrame(data = windy, index = range(len(windy)), columns = ['windy'])


Inputs = pd.concat([outlook,TemperatureAndHumidity,windy],axis = 1)

Play = pd.DataFrame(data = play, index = range(len(play)), columns = ['PLAY'])

SumOf = pd.concat([Inputs,Play],axis = 1)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(Inputs,Play, random_state =2, test_size = 0.25)

from sklearn.linear_model import  LinearRegression

regressor = LinearRegression()
regressor.fit(x_train,y_train)

y_pred = regressor.predict(x_test)

import statsmodels.api as sm

XList = Inputs.iloc[:,:].values
XList = np.array(XList,dtype = float)
model = sm.OLS(Play,XList).fit()





