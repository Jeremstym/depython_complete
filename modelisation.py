# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 12:25:56 2020

Modélisation, plusieurs essais

@author: Jérémie Stym-Popper
"""

###---- Imports----

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from custom_words import super_liste
from vectorize_text_dep import countervect, counter_maker, tfidf_maker
from super_dataframe import gd_df, all_df, LR_df, LFI_df, SOC_df


###---- Création d'un Random Forest Classifier------

super_vectorizer = counter_maker(all_df)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

super_vectorizer = super_vectorizer.reset_index()

y = super_vectorizer['groupe']
X = super_vectorizer.drop('groupe', axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20)

clf1 = RandomForestClassifier()
clf1.fit(X_train, y_train)


y_pred = clf1.predict(X_test)
print(classification_report(y_test, y_pred), clf1.score(X_test,y_test))

###--- Pareil, mais avec tf-idf----

super_vectorizer = tfidf_maker(all_df)

super_vectorizer = super_vectorizer.reset_index()

y = super_vectorizer['groupe']
X = super_vectorizer.drop('groupe', axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20)

clf1 = RandomForestClassifier()
clf1.fit(X_train, y_train)


y_pred = clf1.predict(X_test)
print(classification_report(y_test, y_pred), clf1.score(X_test,y_test))



### -- Autre essai --------


drop = gd_df[gd_df['groupe']=='LR'].sample(65)
gd_df = gd_df.drop(drop.index)
gd_vectorizer = counter_maker(gd_df).reset_index()

y_gd = gd_vectorizer['groupe']
X_gd = gd_vectorizer.drop('groupe', axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X_gd, y_gd, test_size=0.33)

clf1 = RandomForestClassifier()
clf1.fit(X_train, y_train)

from sklearn.metrics import classification_report

y_pred = clf1.predict(X_test)
print(classification_report(y_test, y_pred))

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf1, X_train, y_train, cv=5)
scores.mean()

from sklearn.svm import SVC

clf2 = SVC()
clf2.fit(X_train, y_train)

clf2.predict(X_test[0:2])

y_pred = clf2.predict(X_test)
print(classification_report(y_test, y_pred))




