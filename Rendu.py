# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.7.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] colab_type="text" id="view-in-github"
# <a href="https://colab.research.google.com/github/rturquier/depythons/blob/main/Rendu.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="AyDgk7w54dfJ"
# #### Projet informatique - Python pour le data scientist
# ##### Jérémie Stym-Popper, Luca Teodorescu, Rémi Turquier
# # _La République en Marche_ est-elle de gauche ou de droite ?
#
# [Titre provisoire]

# %% id="u6CCHVOX7XDx"
# Imports
from urllib import request

import numpy as np
import pandas as pd
import spacy

from sklearn.model_selection import train_test_split

# %%
# Si les imports de la cellule suivante ne fonctionnent pas,
# on peut exécuter ces lignes :

# import sys
# directory = r"path/to/repo"
# sys.path.append(directory)

# %%
from depute_api import CPCApi

# %% [markdown] id="H6wrFWawkgh1"
# ## Récupération des données
# Les données ont été récupérées par l'intermédiaire de [l'API](https://github.com/regardscitoyens/nosdeputes.fr/blob/master/doc/api.md) mise à disposition par l'association Regards citoyens.
#
# Nous avons d'abord utilisé un module nommé depute_api, que nous avons ensuite complété avec deux fonctions :
# Les fonctions *interventions* et *interventions2* permettent d'entrer le nom d'un député pour obtenir une liste d'interventions (sous forme de liste de str).

# %% [markdown] id="zGU4ALalPCVS"
# Ensuite, nous avons créé plusieurs DataFrames à l'aide de la fonction interventions 2, avec les fonctions suivantes :

# %% id="eJlvLIZsPCVT"
from get_dep_remi import stockintervention

# %% [markdown] id="TwHj_lLqPCVT"
# Voici un exemple d'utilisation. Pour éviter de perdre du temps ici (la fonction peut mettre du temps à s'exécuter sur les échantillons de grande taille), on exécute la fonction sur un petit parti politique.

# %% id="hI2VdK4aPCVU" outputId="090c474b-ea59-44e4-b677-d10b5c4030c9"
api = CPCApi()
deputies_json = api.parlementaires()
deputies_df = pd.json_normalize(deputies_json)

UAI_df = stockintervention("UAI")
UAI_df


# %% [markdown] id="xM3XXd3KlFz0"
# ## Nettoyage des données
# Après une exploration préliminaire, nous avons choisi de nous concentrer sur trois groupes parlementaires.
# Nous avons sélectionné les interventions des groupes LFI (La France Insoumise) et SOC (Socialistes) pour la gauche, et le groupe LR (Les Républicains) pour la droite.
# ### Création de la variable cible et mise en forme *tidy*
# %% id="im9QOPF57LlX"
### Import des données brutes récupérées avec l'API
data_url = (
    "https://raw.githubusercontent.com/rturquier/depythons/main/Stock_csv/gd2_inter.csv"
)
df_brut = pd.read_csv(data_url)
df_brut.sample(n=5)

# %% colab={"base_uri": "https://localhost:8080/", "height": 204} id="KhTIgGQn73gU" outputId="4ab419d5-647e-44a7-e67d-e2b8ea0d9002"
# Création d'une indicatrice `droite` qui sera la cible de la classification
df_brut = df_brut.assign(droite=df_brut["groupe"] == "LR")


# %% colab={"base_uri": "https://localhost:8080/", "height": 419} id="O0lFEGtugn2k" outputId="1514b6d3-6d2c-4796-c7d1-ba2c2e4122b9"
# Régler un problème de type
from ast import literal_eval


def convert_to_list(interventions):
    return literal_eval(str(interventions))


df_brut["interventions"] = df_brut["interventions"].apply(convert_to_list)

# %% id="K0wHQO2L-MBX"
# Séparer toutes les interventions en colonnes différentes
df_tidy = df_brut.explode("interventions")
df_tidy

# %%
# -- Équilibrage du nombre de d'interventions --
# Nous nous sommes posé la question de l'équilibrage des données.
# Après avoir envisagé plusieurs méthodes, nous avons choisi d'équilibrer les
# donées à la main. Ce n'est pas forcément la meilleure méthode, mais c'est
# sans doute la plus simple.
def n_droite_n_gauche(df):
    count = df.droite.value_counts()
    return count[True], count[False]


def balance_left_right(df):
    n_droite, n_gauche = n_droite_n_gauche(df)
    df = df.sort_values(by=["droite"], ascending=False)

    if n_droite > n_gauche:
        df = df[n_droite - n_gauche :]
    elif n_droite < n_gauche:
        df = df[2 * n_droite :]

    return df


df_tidy = balance_left_right(df_tidy)
n_droite_n_gauche(df_tidy)


# %% [markdown]
# ### Longueur des interventions et regroupement par groupes de 5 interventions

# %%
# Création d'une variable qui contient la longueur des interventions
df_tidy = df_tidy.assign(longueur=df_tidy["interventions"].str.len())

# %%
# Regroupement par groupes de 5 interventions
# Pour cela, on crée une variable `numero_paquet_de_5` qui prend la même valeur
# pour 5 députés du même bord.
n_droite, n_gauche = n_droite_n_gauche(df_tidy)

df_tidy = df_tidy.sort_values(by=["droite"], ascending=False)
df_tidy["numero_paquet_de_5"] = list(range(n_droite)) + list(range(n_gauche))
df_tidy["numero_paquet_de_5"] = np.floor(df_tidy["numero_paquet_de_5"] / 5)
df_tidy["numero_paquet_de_5"] = df_tidy["numero_paquet_de_5"].astype(int)


# %%
# Grouper par bord politique et par numéro de paquet de 5, puis aggréger
df_collapsed = (
    df_tidy.drop(columns=["groupe", "nom"])
    .groupby(["droite", "numero_paquet_de_5"])
    .agg({"interventions": "".join, "longueur": ["min", "max", "mean"]})
    .reset_index()
)

# Arranger le nom des colonnes
df_collapsed.columns = [
    "_".join(col).rstrip("_") for col in df_collapsed.columns.values
]

df_collapsed = df_collapsed.drop(columns="numero_paquet_de_5").rename(
    columns={"interventions_join": "interventions"}
)

df_collapsed

# %% [markdown]
# Il reste à traiter le texte. On applique les transformations vues dans le dernier TD : mettre en minuscule, séparer tous les mots (tokenisation), supprimer les mots courants (stopwords), et ramener à la racine grammticale (lemmatisation).

# %%
# Mise en minuscules
df_spacy = df_collapsed.assign(interventions=df_collapsed.interventions.str.lower())

# %%
# Tokenisation
# Commande pour télécharger les données pour la version française de spaCy :
# python -m spacy download fr_core_news_sm
sp = spacy.load("fr")
df_spacy["interventions"] = df_spacy.interventions.apply(lambda x: sp(x))

# %%
# Lemmatisation
df_spacy["interventions"] = df_spacy.interventions.apply(
    lambda tokens: [token.lemma_ for token in tokens]
)

# %%
# Stopwords
stop_words = sp.Defaults.stop_words | {"'", ",", ";", ":", " "}

df_spacy["interventions"] = df_spacy.interventions.apply(
    lambda words: [word for word in words if not word in stop_words]
)

# %%
# Résultat
print(
    str(df_collapsed.interventions[42]) + "\n ---> \n" + str(df_spacy.interventions[42])
)

# %% [markdown]
# Maintenant que le traitement préparatoire des données est terminé, nous
# pouvons passer aux parties exploration et modélisation. Pour éviter toute
# fuite d'information des données de test vers les données d'entrainement,
# nous faisons dès maintenant la séparation. Nous avons retenu la proportion
# *train* / *test* usuelle de 80% / 20%.

## %%
X = df_spacy.drop("droite", axis=1)
y = df_spacy["droite"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

train = pd.merge(X_train, y_train, left_index=True, right_index=True)
test = pd.merge(X_test, y_test, left_index=True, right_index=True)


# %% [markdown] id="p0ZAydeEPCVc"
# ## Analyse descriptive des données
#
# Nous allons maintenant dans cette partie visualiser les tendances dans les différents partis, ainsi que les mots qui sont les plus utilisés. Nous commençons par le wordcloud.
# Avant toutes choses, nous importons et complétons un texte de stopwords pour retirer tous les mots impertinents de la visualisation, et de la modélisation à suivre.

# %%
# Création d'un stopwords

stopping_list = request.urlopen(
    "https://raw.githubusercontent.com/rturquier/depythons/main/stopwords-fr.txt"
).read()
stopping_list = stopping_list.decode("utf-8")
stopwords_list = stopping_list.split("\n")

# %% id="AhEb0zObPCVd"
from wordcloud_depython import wordcloud_gen

# %% [markdown] id="esjyOFUXPCVe"
# Quelques petits essais...

# %% id="OU4eFFXUPCVe" outputId="ab16e9fe-7214-45f6-e213-407f17829ccd"
wordcloud_gen("Jean-Luc Mélenchon"), wordcloud_gen("Eric Ciotti")

# %% [markdown] id="O-RmXt3xPCVf"
# ### Liste de mots customisée
#
# Nous avons créé deux fonction qui permettent de retourner les mots les plus utilisés par les membres d'un parti. Cette liste de mots va nous servir pour modéliser les champs lexicaux (quel parti a le plus tendance à utiliser tel mot ?). Nous créons cette liste de 144 mots sous le nom de *super_liste*.

# %%
from custom_words import super_liste

# %% [markdown] id="I0lSH6JWPCVh" outputId="a243f78a-7908-4342-b754-89bfd768da17"
# # Modéslisation
#
# Nous passons maintenant à la partie de la modélisation. Nous avons pour cela procédé en plusieurs espace.
# 1. La première consiste en la création d'une table *df_simple* qui regroupe uniquement les partis de gauche classiques (Socisalites et LFI) et le parti LR. Nous rajoutons une colonne "droite" qui renvoie **True** si le parti est de droite (LR ici), **False** sinon. Cela permettra après d'entraîner le modèle supervisé, le label étant donné par cette colonne.
# 2. La deuxième étape consiste à transfomrer le DataFrame contenant "Parti politique", "Nom du député" et "Interventions" en une matrice **TF-IDF**. Les détails seront donnés plus bas.
# 3. La deuxième étape consiste à entraîner deux modèles (ici, **RandomForestClassifier** et **SVC**) sur les données, en évaluant à chaque fois quels sont les meileurs hyperparamètres à l'aide de la méthode de validation croisée.
# 4. La dernière étape consiste à exécuter le modèle sur les députés du parti LREM pour prédire à quel bord politique ils pourraient potentiellement appartenir, commte tenu des mots les plus courants de leurs interventions.

# %% [markdown]
# #### Première étape
# Nous créons d'abord la matrice df_simple

# %%
data_url = (
    "https://raw.githubusercontent.com/rturquier/depythons/main/Stock_csv/gd2_inter.csv"
)
df_brut = pd.read_csv(data_url)
df_brut.sample(n=5)

# Création d'une indicatrice `droite` qui sera la cible de la classification
df_brut = df_brut.assign(droite=df_brut["groupe"] == "LR").sort_values(
    by=["droite"], ascending=False
)
df_brut.head()

df_simple = df_brut.assign(droite=df_brut["groupe"] == "LR")
df_simple

# %% [markdown]
# #### Deuxième étape
#
# Nous allons maintenant créer les matrices **Tf-Idf** qui vont nous servir pour les modèles.

# %%
# Voici tous les imports qui sont nécessaires pour cette partie et la suite

import collections

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import GridSearchCV

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

LREM_df = pd.read_csv(
    "https://raw.githubusercontent.com/rturquier/depythons/main/Stock_csv/LREM2_inter.csv"
)

# %% [markdown]
# Nous créons une matrice td-idf pour le DataFrame gd_df, puis nous séparons les lignes en un échantillon d'entraînement et un échantillon de test.
# Ensuite nous effectuons la même chose avec le DataFrame df_simple.

# %%
pipe_idf = make_pipeline(CountVectorizer(vocabulary=super_liste), TfidfTransformer())

tf_idf_simple = pipe_idf2.fit_transform(
    df_simple["interventions"].values.astype("U")
).toarray()

simple_features = pd.DataFrame(tf_idf_simple, index=df_simple["droite"])
simple_features = simple_features.reset_index()  # Pour que 'groupe' devienne la target

y = simple_features["droite"]
X = simple_features.drop("droite", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
simple_features

# %% [markdown]
# #### Troisième étape
# Nous allons maintenant évaluer quels sont les meilleurs hyperparamètres pour chaque modèle.
# * D'abord le modèle RandomForestClassifier
# * Ensuite le modèle SVC

# %%
###---- Évaluation des hyperparamètres avec validation croisée pour le RandomForestClassifier -----
param_grid = {
    "n_estimators": [200, 500],
    "max_features": ["auto", "sqrt", "log2"],
    "max_depth": [4, 5, 6, 7, 8],
    "criterion": ["gini", "entropy"],
}

CV_rfc2 = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5)

CV_rfc2.fit(X_train, y_train)
opti_param1 = CV_rfc2.best_params_

"""
La fonction met du temps à s'exécuter. Voici les paramètres qu'elle trouve
{'criterion': 'gini',
 'max_depth': 8,
 'max_features': 'auto',
 'n_estimators': 200}
Pas la peine de la relancer à chaque fois.
"""

# %%
## Validation croisée pour trouver hyperparamètres pour le modèle SVC -----

tuned_parameters = [
    {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
    {"kernel": ["linear"], "C": [1, 10, 100, 1000]},
]


CV_rfc3 = GridSearchCV(estimator=SVC(), param_grid=tuned_parameters, cv=5)

CV_rfc3.fit(X_train, y_train)
opti_param2 = CV_rfc3.best_params_
print("Les meilleurs hyperparamètres sont " + str(opti_param2))

# %% [markdown]
# Nous allons maintenant entraîner les deux modèles successivment et évaluer leur pertinence.

# %%
clf = RandomForestClassifier(
    criterion="gini", max_depth=8, max_features="auto", n_estimators=200
)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(classification_report(y_test, y_pred))
print("Le score du test est " + str(clf.score(X_test, y_test)))

# %%
svc_try = SVC(C=10, kernel="linear")
svc_try.fit(X_train, y_train)

y_pred2 = svc_try.predict(X_test)
print(classification_report(y_test, y_pred2))
print("Le score du test est " + str(svc_try.score(X_test, y_test)))

# %% [markdown]
# #### Quatrième étape
# Nous regardons maintenant quel classfication effectue le modèle sur le parti LREM.
# Premièrement, nous transformons d'abord la matrice LREM.

# %%
final_pipe = make_pipeline(CountVectorizer(vocabulary=super_liste), TfidfTransformer())

tf_idf_LREM = final_pipe.fit_transform(
    LREM_df["interventions"].values.astype("U")
).toarray()

LREM_features = pd.DataFrame(tf_idf_LREM)

# %%
LREM_pred = clf.predict(LREM_features)
print(collections.Counter(LREM_pred))
print(
    "Le modèle RFC classe",
    str(collections.Counter(LREM_pred)[1]),
    "députés à droite et",
    str(collections.Counter(LREM_pred)[0]),
    "à gauche",
)

# %%
LREM_pred2 = svc_try.predict(LREM_features)
print(collections.Counter(LREM_pred2))
print(
    "Le modèle SVC classe",
    str(collections.Counter(LREM_pred2)[1]),
    "députés à droite et",
    str(collections.Counter(LREM_pred2)[0]),
    "à gauche",
)
