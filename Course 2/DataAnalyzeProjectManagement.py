# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 20:10:05 2020

@author: ABRA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('veriler.csv')

from sklearn.impute import SimpleImputer


Yas = veriler.iloc[:,1:4].values

imputer =SimpleImputer(missing_values=np.nan,strategy='mean')

imputer = imputer.fit(Yas[:,1:4])
Yas[:,1:4] = imputer.transform(Yas[:,1:4])


ulke = veriler.iloc[:,0:1].values


from sklearn import preprocessing

le = preprocessing.LabelEncoder()

ulke[:,0] = le.fit_transform(veriler.iloc[:,0])


ohe = preprocessing.OneHotEncoder()
ulke = ohe.fit_transform(ulke).toarray()
print(ulke)

sonuc = pd.DataFrame(data = ulke, index = range(22),columns = ['fr','tr','us'])
print(sonuc)

sonuc2 = pd.DataFrame(data = Yas, index = range(22),columns = ['boy','kilo','yas'])
print(sonuc2)

cinsiyet = veriler.iloc[:,-1].values

sonuc3 = pd.DataFrame(data = cinsiyet,index = range(22),columns = ['cinsiyet'])
print(sonuc3)
print()


s = pd.concat([sonuc,sonuc2],axis = 1)
print(s)


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(s, sonuc3, test_size = 0.33,random_state = 0)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

























