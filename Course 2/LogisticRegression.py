# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 20:10:05 2020

@author: ABRA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

veriler = pd.read_csv('veriler.csv')

x = veriler.iloc[6:,1:4].values
y = veriler.iloc[6:,4:].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.33,random_state = 0)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)


from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state=0)
logr.fit(X_train, y_train)

y_pred = logr.predict(X_test)

from sklearn.metrics import confusion_matrix

con_matrix = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score

score = accuracy_score(y_test,y_pred)
print(score)













