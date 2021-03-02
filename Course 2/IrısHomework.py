
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

veriler = pd.read_excel('Iris.xls')

x = veriler.iloc[:,:-1]
y = veriler.iloc[:,-1:]

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.33,random_state = 0)


from sklearn.preprocessing import StandardScaler

sc = StandardScaler()

X_train = sc.fit_transform(x_train)
X_test = sc.transform(x_test)

#Random Forest
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(random_state=1,n_estimators=10, criterion="gini")

rfc.fit(X_train, y_train)
y_pred = rfc.predict(X_test)

from sklearn.metrics import confusion_matrix
RF_cm = confusion_matrix(y_test,y_pred)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(criterion="entropy")
dtc.fit(X_train,y_train)

#Naive Bayes
from sklearn.naive_bayes import MultinomialNB

y_pred = dtc.predict(X_test)
DT_cm = confusion_matrix(y_test,y_pred)

gnb = MultinomialNB()
gnb.fit(x_train, y_train)

y_pred = gnb.predict(x_test)

NB_cm = confusion_matrix(y_test,y_pred)

#SVC
from sklearn.svm import SVC

svc = SVC(kernel = 'linear')
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

from sklearn.metrics import confusion_matrix
SVC_cm = confusion_matrix(y_test,y_pred)

#K-NN
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=1,metric = 'minkowski')
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

from sklearn.metrics import confusion_matrix

KNN_cm = confusion_matrix(y_test,y_pred)

#LogisticRegression
from sklearn.linear_model import LogisticRegression

logr = LogisticRegression(random_state=0)
logr.fit(X_train, y_train)

y_pred = logr.predict(X_test)

from sklearn.metrics import confusion_matrix

logr_cm= confusion_matrix(y_test, y_pred)








