# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 20:10:05 2020

@author: ABRA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('Lesson 2\\satislarveriler.csv')

print(veriler)

# from sklearn.impute import SimpleImputer

aylar = veriler[['Aylar']]
print(aylar)

satislar = veriler[['Satislar']]
print(satislar)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(aylar, satislar, test_size = 0.33,random_state = 1)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

#X_train = sc.fit_transform(x_train)
#X_test = sc.fit_transform(x_test)

#Y_train = sc.fit_transform(y_train)
#Y_test = sc.fit_transform(y_test)

from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(x_train, y_train)

tahmin = lr.predict(x_test)
print(tahmin)

x_train = x_train.sort_index()
y_train = y_train.sort_index()

plt.plot(x_train, y_train)
plt.plot(x_test,lr.predict(x_test))

plt.title("Aylara Göre Satış")
plt.xlabel("Aylar")
plt.ylabel("Satışlar")
















