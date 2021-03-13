
import pandas as pd
import numpy as np

#Preprocssing
file = 'Restaurant_Reviews.csv'

data = []
Y = []
X = []

with open(file, 'r') as csvfile:
    for line in csvfile.readlines():
        data.append(line)

data = np.array(data)[1:]

for i in data:
    i = i.replace('"','')
    Y.append(i[-2])
    X.append(i[:-3])

X = pd.DataFrame(data = X, index = range(len(X)), columns=['Review'])
Y = pd.DataFrame(data = Y, index = range(len(X)), columns=['Liked'], dtype = np.int32)
data = pd.concat([X,Y], axis = 1)

import re
import nltk
nltk.download('stopwords')

from nltk.stem.porter import PorterStemmer

ps = PorterStemmer()

from nltk.corpus import stopwords

corpus = []
for i in range(len(data)):
    review = re.sub('[^a-zA-Z]',' ',data['Review'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)
    
#Feature Extraction with Bag of Words aka BOW
from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 2000)

Inputs = cv.fit_transform(corpus).toarray()
Output = data.iloc[:,1].values

#Prediction
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(Inputs, Output, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(x_train,y_train)
y_pred = gnb.predict(x_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)























