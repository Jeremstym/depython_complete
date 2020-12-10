# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 12:25:56 2020

Feature engeenering avec CountVectorizer, Pipeline, et HashingVectorizer
Modélisation, plusieurs essais.
Évaluation et choix des hyperparamètres


@author: Jérémie Stym-Popper
"""

###---- Imports----

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from custom_words import super_liste
from vectorize_text_dep import  give_lotofwords
from super_dataframe import gd_df, all_df, LR_df, LFI_df, SOC_df


###---- Création d'un échantillon pour le modèkle ------

counter = CountVectorizer(vocabulary=super_liste).fit_transform(give_lotofwords(gd_df)).toarray()
weak_features = pd.DataFrame(counter, index=gd_df['groupe'])
weak_features = weak_features.reset_index() # Pour que 'groupe' devienne la target

y = weak_features['groupe']
X = weak_features.drop('groupe', axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20)

###---- Évaluation des hyperparamètres -----

param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

CV_rfc = GridSearchCV(estimator=RandomForestClassifier(), 
                      param_grid=param_grid, cv=5)

CV_rfc.fit(X_train, y_train)
opti_param = CV_rfc.best_params_

"""
La fonction met un temps fou à s'exécuter. Voici les paramètres qu'elle trouve
{'criterion': 'gini',
 'max_depth': 7,
 'max_features': 'auto',
 'n_estimators': 200}
Pas la peine de la relancer à chaque fois.
"""

#--- Modélisation ! -------------

clf1 = RandomForestClassifier(criterion="gini", max_depth=7,
                              max_features="auto", n_estimators=200)
clf1.fit(X_train, y_train)


y_pred = clf1.predict(X_test)
print(classification_report(y_test, y_pred), clf1.score(X_test,y_test))



###--- Pareil, mais avec tf-idf -----------------------
#---- Création d'un échantillon pour le modèle -------------


pipe_idf = make_pipeline(CountVectorizer(vocabulary=super_liste),
                         TfidfTransformer())

tf_idf_result = pipe_idf.fit_transform(give_lotofwords(gd_df)).toarray()

nos_features = pd.DataFrame(tf_idf_result, index=gd_df['groupe'])
nos_features = nos_features.reset_index() # Pour que 'groupe' devienne la target

y2 = nos_features['groupe']
X2 = nos_features.drop('groupe', axis=1)

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X2, y2, test_size=0.20)



###---- Évaluation des hyperparamètres -----


param_grid = { 
    'n_estimators': [200, 500],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth' : [4,5,6,7,8],
    'criterion' :['gini', 'entropy']
}

CV_rfc2 = GridSearchCV(estimator=RandomForestClassifier(), 
                      param_grid=param_grid, cv=5)

CV_rfc2.fit(X_train2, y_train2)
opti_param2 = CV_rfc2.best_params_

"""
La fonction met un temps fou à s'exécuter. Voici les paramètres qu'elle trouve
{'criterion': 'gini',
 'max_depth': 8,
 'max_features': 'auto',
 'n_estimators': 200}
Pas la peine de la relancer à chaque fois.
"""
# ---- Modélisation ! --------------------

clf2 = RandomForestClassifier(criterion="gini", max_depth=8,
                              max_features="auto", n_estimators=200)

clf2.fit(X_train2, y_train2)


y_pred2 = clf2.predict(X_test2)
print(classification_report(y_test2, y_pred2), clf2.score(X_test2,y_test2))


### -- Autre essai brouillons --------
"""

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
"""



