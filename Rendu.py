# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent,md
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
# # Peut-on prédire l'appartenance politique selon le discours des députés ?
#
#
# Ce projet consiste à effectuer du *Natural Language Processing* sur le discours que tiennent les députés parlementaires.
#
# L'objectif principal consiste à entraîner un ou plusieurs modèles sur les députés appartenant à des partis étant traditionnellement reconnus de gauche (PS, LFI) et de droite (LR), puis essayer de prédire cette classfication sur le parti LREM qui se situe entre les deux bords.
#
# Nous avons prélevé les textes depuis deux sources : une première manière à consister en utilisant les interventions des députés à l'Assemblée (nous avons utiliser l'API venant du site [nosdeputes.fr](https://www.nosdeputes.fr/), puis scrappé les discours d'un député ciblé. Nous avons également scrappé Twitter pour retrouver les tweets des députés et diversifier ainsi nos sources.
#
# Dans ce Notebook, nous utilisons à plusieurs reprises des modules que nous avons codés par ailleurs et déposés sur Github. Ces modules utilisent des packages comme **wordcloud**, **unidecode**, **warnings** et **fuzzywuzzy** qu'il faudrait préalablement installer pour pouvoir tout lire.

# %% id="u6CCHVOX7XDx"
# Imports
from urllib import request

import numpy as np
import pandas as pd
import spacy

import collections
import re
import matplotlib.pyplot as plt


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
# Le nombre d'interventions est bien équilibré.
#
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
sp = spacy.load("fr_core_news_sm")
df_spacy["interventions"] = df_spacy.interventions.apply(lambda x: sp(x))

# %%
# Lemmatisation
df_spacy["interventions"] = df_spacy.interventions.apply(
    lambda tokens: [token.lemma_ for token in tokens]
)
df_zipf = df_spacy.copy()
# %%
# Stopwords
stop_words = sp.Defaults.stop_words | {"'", ",", ";", ":", " ", "", "."}

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
#


# %%
from sklearn.model_selection import train_test_split

X = df_spacy.drop("droite", axis=1)
y = df_spacy["droite"]

# L'argument `random_state` permet d'obtenir des résultats reproductibles.
split_list = train_test_split(
     X, y, test_size=0.2, random_state=42
)

# Réindexer maintenant pour éviter des soucis au moment du tf-idf
X_train, X_test, y_train, y_test = [
    df.reset_index().drop(columns = ["index"]) for df in split_list
]

# %% [markdown] id="p0ZAydeEPCVc"
# # Analyse descriptive des données (Visualisation)
#
# Nous allons maintenant dans cette partie visualiser les tendances dans les différents partis, ainsi que les mots qui sont les plus utilisés. Nous commençons par la loi Zipf, en regardant quels mots sont les plus employés par les partis de droite et de gauche.
#
# On essaye de faire une analyse des fréquences des mots selon le catégorie droite/gauche et vérifier une potentielle loi de Zipf
# Dans un premier temps sans enlever les stopwords
#
#

# %%
# On sépare en deux dataframe une pour chaque catégorie et on fait de même avec la dataframe spacy travaillée juste avant
df_zipf_droite = df_zipf[df_zipf["droite"]]
df_zipf_gauche = df_zipf[df_zipf["droite"] != True]
df_spacy_droite = df_spacy[df_spacy["droite"]]
df_spacy_gauche = df_spacy[df_spacy["droite"] != True]
# %%
# Dictionnaries de wordcount
wordcount_droite = collections.defaultdict(int)
wordcount_gauche = collections.defaultdict(int)
for inters in df_zipf_droite["interventions"]:
    for word in inters:
        wordcount_droite[word] += 1
for inters in df_zipf_gauche["interventions"]:
    for word in inters:
        wordcount_gauche[word] += 1

# %%
# On va afficher les 20 mots les plus populaires pour la gauche et la droite en comptant tous les mots
fig, axs = plt.subplots(2, 1, figsize=(15, 10))
fig.suptitle("Fréquences d'utilisation des mots dans les allocutions pour chaque bords politiques")
mcg = sorted(wordcount_gauche.items(), key=lambda k_v: k_v[1], reverse=True)[:20]
mcg = dict(mcg)
namesg = list(mcg.keys())
valuesg = list(mcg.values())
axs[0].bar(range(len(mcg)),valuesg,tick_label=namesg, color='red')
axs[0].set_title('Pour la gauche')
mcd = sorted(wordcount_droite.items(), key=lambda k_v: k_v[1], reverse=True)[:20]
mcd = dict(mcd)
namesd = list(mcd.keys())
valuesd = list(mcd.values())
axs[1].bar(range(len(mcd)),valuesd,tick_label=namesd, color='blue')
axs[1].set_title('Pour la droite :')
# %%
# On va maintenant voir le résultat avec une liste de stopwords
wordcount_droite = collections.defaultdict(int)
wordcount_gauche = collections.defaultdict(int)
for inters in df_spacy_droite["interventions"]:
    for word in inters:
        word = re.sub(r"\W", "", word)
        if word not in stop_words:
            wordcount_droite[word] += 1
for inters in df_spacy_gauche["interventions"]:
    for word in inters:
        word = re.sub(r"\W", "", word)
        if word not in stop_words:
            wordcount_gauche[word] += 1



# %%
# On va afficher les 10 mots les plus populaires pour la gauche et la droite
fig, axs = plt.subplots(2, 1, figsize=(20, 10))
fig.suptitle("Fréquences d'utilisation des mots dans les allocutions pour chaque bords politiques sans stopwords,")
mcg = sorted(wordcount_gauche.items(), key=lambda k_v: k_v[1], reverse=True)[:20]
mcg = dict(mcg)
namesg = list(mcg.keys())
valuesg = list(mcg.values())
axs[0].bar(range(len(mcg)),valuesg,tick_label=namesg, color='red')
axs[0].set_title('Pour la gauche')
mcd = sorted(wordcount_droite.items(), key=lambda k_v: k_v[1], reverse=True)[:20]
mcd = dict(mcd)
namesd = list(mcd.keys())
valuesd = list(mcd.values())
axs[1].bar(range(len(mcd)),valuesd,tick_label=namesd, color='blue')
axs[1].set_title('Pour la droite :')


# %%
# On a fait le même travail dans visualisation_twitter.py sur une base de donnée twitter scrappé avec notre fichier twitter.py à titre de comparaison
from IPython.display import Image
Image(filename='Tweets_Frequence.png')


# %% [markdown]
#
# Nous allons maintenant visualiser les mots les plus utilisés de manière plus intuitive. À l'aide du module **wordcloud**, il est possible de visualiser quels sont les mots les plus utilisés par un député.

# %%
# Création d'un stopwords à partir d'un fichier txt. téléchargé et complété.

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


# %% [markdown] id="I0lSH6JWPCVh" outputId="a243f78a-7908-4342-b754-89bfd768da17"
# # Modélisation
#
# Nous passons maintenant à la partie modélisation. Pour cela, nous utilisons
# les données que nous avons manipulées jusqu'ici. Nous commençons d'abord par
# du **Features Engineering** en créant une matrice *TF-IDF* pour entraîner
# les modèles.
#
# Les deux modèles que nous avons choisis sont **Random Forest Classifier** et
# **SVC**. Ils vont nous permettre de comparer les résultats et les scores
# obtenus.
#
# À chaque fois, nous nous effectuons une **validation croisée** pour déterminer
# quels sont les meilleurs hyperparamètres, avant d'entaîner les modèles.
#
# Enfin, nous finissons par utiliser les modèles pour prédire à quel bord
# politique appartiennent les députés LREM.


# %%
# Voici tous les imports qui sont nécessaires pour cette partie et la suite

import collections

from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

LREM_df = pd.read_csv(
    "https://raw.githubusercontent.com/rturquier/depythons/main/Stock_csv/LREM2_inter.csv"
)



# %% [markdown]
# **Définition du Tfidf**
# Tfidf est une manière de transformer un corpus de texte en features. Chaque
# mot a d'autant plus de poids qu'il est relativement fréquent dans un paquet
# de 5 interventions par rapport aux autres paquet. Par exemple, un mot qui
# apparait souvent, mais dans uniformément n'a pas beaucoup d'importance.
#
# Comme la partie IDF (*Inverse Document Frequence*) utilise de l'information
# de tout le corpus, il était important de séparer les données d'entrainement
# des données de test avant d'appliquer tf-idf, dans cette [question StackOverflow](https://stackoverflow.com/questions/47778403/computing-tf-idf-on-the-whole-dataset-or-only-on-training-data).
#
# Nous nous sommes rendu compte qu'utiliser *tous* les mots des interventions
# (hormis les *stopwords*) conduisait à un sur-apprentissage. Cela est du au
# fait que nous avons un nombre de données inférieur au nombre potentiel de
# features créées par tf-idf.
#
# Nous avons testé deux façons de limiter le nombre de features :
# 1. créer une liste de mots limitée à la main, et utiliser l'argument `vocabulary` de `TfidfVectorizer`
# 2. utiliser le paramètre `max_features`
#
# Le code qui a servi à la première approche se trouve dans le fichier
# `custom_words.py`. Nous avons finalement retenu la seconde approche, qui
# tire meilleur parti des outils de NLP.
#

# %%
# On crée d'abord une fonction vide, qui permet de dire au `TfidfVectorizer`
# que nous avons déjà pré-traité le texte.

def dummy_fun(doc):
    return doc

tf_idf = TfidfVectorizer(
    analyzer = 'word',
    tokenizer = dummy_fun,
    preprocessor = dummy_fun,
    token_pattern = None,
    # vocabulary = super_liste
    max_features = 150
)


# %%
# Création des features avec tf-idf
X_train_tf_idf = tf_idf.fit_transform(X_train["interventions"])

# Quelques features retenues :
tf_idf.get_feature_names()[::10]

# %%
# Ajout des features tf_idf à la longueur des interventions
X_train_tf_idf = pd.DataFrame.sparse.from_spmatrix(X_train_tf_idf)
X_train = (X_train.merge(X_train_tf_idf, left_index = True, right_index = True)
            .drop(columns = ["interventions"])
)

# Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

# %% [markdown]
# #### Troisième étape
# Nous allons maintenant évaluer quels sont les meilleurs hyperparamètres pour chaque modèle.
# * D'abord le modèle RandomForestClassifier
# * Ensuite le modèle SVC

# %%
### Validation croisée pour le RandomForestClassifier
param_grid_rfc = {
    "n_estimators": [200, 500],
    "max_features": ["auto", "sqrt", "log2"],
    "max_depth": [4, 5, 6, 7, 8],
    "criterion": ["gini", "entropy"],
}

CV_rfc = GridSearchCV(estimator=RandomForestClassifier(), param_grid=param_grid, cv=5)

CV_rfc.fit(X_train, y_train)
opti_param_rfc = CV_rfc.best_params_

"""
La fonction met du temps à s'exécuter. Voici les paramètres qu'elle trouve
{'criterion': 'gini',
 'max_depth': 6,
 'max_features': 'sqrt',
 'n_estimators': 200}
Pas la peine de la relancer à chaque fois.
"""

# %%
## Validation croisée pour trouver les hyperparamètres pour le modèle SVC

param_grid_svc = [
    {"kernel": ["rbf"], "gamma": [1e-3, 1e-4], "C": [1, 10, 100, 1000]},
    {"kernel": ["linear"], "C": [1, 10, 100, 1000]},
]


CV_svc = GridSearchCV(estimator=SVC(), param_grid=param_grid_svc, cv=5)

CV_svc.fit(X_train, y_train)
opti_param_svc = CV_svc.best_params_
print("Les meilleurs hyperparamètres sont " + str(opti_param_svc))

# %% [markdown]
# Nous allons maintenant entraîner les deux modèles successivment.

# %%
# Mettre `y_train` au bon format
y_train = y_train.values.ravel()

# Entrainement de la forêt aléatoire
clf = RandomForestClassifier(
    criterion="gini", max_depth=6, max_features="sqrt", n_estimators=200
)
clf.fit(X_train, y_train)

# Entrainement de la machine à vecteur de support (SVC)
svc = SVC(C=1000, gamma = 0.0001, kernel="rbf")
svc.fit(X_train, y_train)

# %% [markdown]
# Avant de pouvoir évaluer l'erreur de généralisation, il faut appliquer
# quelques transformations aux données de test.
# Ici, on applique uniquement la méthode `transform` des Transformers, la
# méthode `fit` étant réservée à l'étape d'entrainement.

# %%
# tf-idf
X_test_tf_idf = tf_idf.transform(X_test["interventions"])

X_test_tf_idf = pd.DataFrame.sparse.from_spmatrix(X_test_tf_idf)
X_test = (X_test.merge(X_test_tf_idf, left_index = True, right_index = True)
            .drop(columns = ["interventions"])
)

# Scaling
X_test = scaler.transform(X_test)



# %% [markdown]
# Nous pouvons maintenant appliquer nos modèles aux données de test, et évaluer
# leur erreur de généralisation.

# %%
y_pred_clf = clf.predict(X_test)

print(classification_report(y_test, y_pred_clf))
print("Le score du test est " + str(clf.score(X_test, y_test_clf)))

# %%


y_pred_svc = svc.predict(X_test)
print(classification_report(y_test, y_pred_svc))
print("Le score du test est " + str(svc.score(X_test, y_test_svc)))

# %% [markdown]
# Le


# %% [markdown]
# ##
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
LREM_pred2 = svc.predict(LREM_features)
print(collections.Counter(LREM_pred2))
print(
    "Le modèle SVC classe",
    str(collections.Counter(LREM_pred2)[1]),
    "députés à droite et",
    str(collections.Counter(LREM_pred2)[0]),
    "à gauche",
)
