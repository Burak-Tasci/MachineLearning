import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Gaussion reel sayılar
#multinomial çoklu seçimler
#bernoulli binary seçimler


veriler = pd.read_csv('veriler.csv')

x = veriler.iloc[:,1:4].values
y = veriler.iloc[:,4:].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.33,random_state = 0)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

from sklearn.naive_bayes import GaussianNB, BernoulliNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print("Gaussian NB \n",cm)

gnb = BernoulliNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)



cm = confusion_matrix(y_test,y_pred)
print("Bernoulli NB \n",cm)


gnb = MultinomialNB()
gnb.fit(x_train, y_train)

y_pred = gnb.predict(x_test)



cm = confusion_matrix(y_test,y_pred)
print("Mulitnomial NB \n",cm)































































