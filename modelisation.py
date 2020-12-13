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
import collections

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

from custom_words import super_liste
from super_dataframe import gd_df, LREM_df
from one_line_df import df_simple


###---- Création d'un échantillon pour le modèle ------

counter = CountVectorizer(vocabulary=super_liste).fit_transform(gd_df['interventions'].values).toarray()
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

tf_idf_result = pipe_idf.fit_transform(gd_df['interventions'].values).toarray()

nos_features = pd.DataFrame(tf_idf_result, index=gd_df['groupe'])
nos_features = nos_features.reset_index() # Pour que 'groupe' devienne la target

y2 = nos_features['groupe']
X2 = nos_features.drop('groupe', axis=1)

X_train2, X_test2, y_train2, y_test2 = train_test_split(
    X2, y2, test_size=0.20)



###---- Évaluation des hyperparamètres avec validation croisée -----


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


###--- Td_idf avec df_simple -----------------------
#---- Création d'un échantillon pour le modèle -------------


pipe_idf2 = make_pipeline(CountVectorizer(vocabulary=super_liste),
                         TfidfTransformer())

tf_idf_tidy = pipe_idf2.fit_transform(df_simple['interventions'].values.astype('U')).toarray()

tidy_features = pd.DataFrame(tf_idf_tidy, index=df_simple['droite'])
tidy_features = tidy_features.reset_index() # Pour que 'groupe' devienne la target

y3 = tidy_features['droite']
X3 = tidy_features.drop('droite', axis=1)

X_train3, X_test3, y_train3, y_test3 = train_test_split(
    X3, y3, test_size=0.20)


#### Modélisation avec RandomForestClassifier -----


clf3 = RandomForestClassifier(criterion="gini", max_depth=8,
                              max_features="auto", n_estimators=200)

clf3.fit(X_train3, y_train3)


y_pred3 = clf3.predict(X_test3)

print(classification_report(y_test3, y_pred3), clf3.score(X_test3, y_test3))


### ---- Modélisation avec SVC, avec les mêmes features ------
## Validation croisée pour trouver hyperparamètres -----

tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-3, 1e-4],
                     'C': [1, 10, 100, 1000]},
                    {'kernel': ['linear'], 'C': [1, 10, 100, 1000]}]


CV_rfc3 = GridSearchCV(estimator=SVC(), 
                      param_grid=tuned_parameters, cv=5)

CV_rfc3.fit(X_train3, y_train3)
opti_param3 = CV_rfc3.best_params_
print("Les meilleurs hyperparamètres sont " + str(opti_param3))

#--- Modélisation ! ---------------

svc_try = SVC(C=10, kernel="linear")
svc_try.fit(X_train3, y_train3)

y_pred4 = svc_try.predict(X_test3)
print(classification_report(y_test3, y_pred4)) 
print("Le score du test est " + str(clf3.score(X_test3, y_test3)))

## On va utiliser ce dernier modèle pour prévoir l'appartenance politique 
# des députés LREM.

final_pipe = make_pipeline(CountVectorizer(vocabulary=super_liste),
                         TfidfTransformer())

tf_idf_LREM = final_pipe.fit_transform(LREM_df['interventions'].values.astype('U')).toarray()

LREM_features = pd.DataFrame(tf_idf_LREM)

LREM_pred = svc_try.predict(LREM_features)
LREM_pred

collections.Counter(LREM_pred)


### -- Autres essais brouillons, ne pas faire attention --------
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



