# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 21:25:19 2022

@author: tom97
"""

import pandas as pd

messages = pd.read_csv('SMSSpamCollection',sep = '\t',names = ['class','message'])


import re
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

lemmatizer = WordNetLemmatizer()

sen = []

for i in range(len(messages)):
    review = re.sub('[^a-zA-Z]',' ',messages['message'][i])
    review = review.lower()
    review = review.split()
    review = [lemmatizer.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    sen.append(review)
    
    
from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features = 3000)
X = cv.fit_transform(sen).toarray()

Y = pd.get_dummies(messages['class'],drop_first = True)

from sklearn.model_selection import train_test_split

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.25,random_state = 0)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB().fit(X_train,Y_train)

Y_pred = model.predict(X_test)



from sklearn.metrics import confusion_matrix, accuracy_score

cm = confusion_matrix(Y_test, Y_pred)
acc = accuracy_score(Y_test,Y_pred)