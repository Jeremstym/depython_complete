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
#     display_name: depythons
#     language: python
#     name: depythons
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
import numpy as np
import pandas as pd

# %% [markdown] id="H6wrFWawkgh1"
# ## Récupération des données
# Les données ont été récupérées par l'intermédiaire de [l'API](https://github.com/regardscitoyens/nosdeputes.fr/blob/master/doc/api.md) mise à disposition par l'association Regards citoyens.
#
# Nous avons d'abord utilisé un module nommé depute_api, que nous avons ensuite complété avec trois fonctions :
# Les fonctions *interventions* et *interventions2* permettent d'entrer le nom d'un député pour obtenir une liste d'interventions (sous forme de liste de str).
# La fonction ...

# %% id="OW5P37oZPCVR"
# ----- Codage de l'API --------------------------------------------------
from operator import itemgetter
import requests
import warnings
import re
import bs4
import unidecode
from urllib import request


with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=UserWarning)
    from fuzzywuzzy.process import extractBests


__all__ = ["CPCApi"]


def memoize(f):
    cache = {}

    def aux(*args, **kargs):
        k = (args, tuple(sorted(kargs.items())))
        if k not in cache:
            cache[k] = f(*args, **kargs)
        return cache[k]

    return aux


class CPCApi(object):
    format = "json"

    def __init__(self, ptype="depute", legislature=None):
        """
        type: depute or senateur
        legislature: 2007-2012 or None
        """

        assert ptype in ["depute", "senateur"]
        assert legislature in ["2007-2012", "2012-2017", None]
        self.legislature = legislature
        self.ptype = ptype
        self.ptype_plural = ptype + "s"
        self.base_url = "https://%s.nos%s.fr" % (
            legislature or "www",
            self.ptype_plural,
        )

    def synthese(self, month=None):
        """
        month format: YYYYMM
        """
        if month is None and self.legislature == "2012-2017":
            raise AssertionError(
                "Global Synthesis on legislature does not work, see https://github.com/regardscitoyens/nosdeputes.fr/issues/69"
            )

        if month is None:
            month = "data"

        url = "%s/synthese/%s/%s" % (self.base_url, month, self.format)

        data = requests.get(url).json()
        return [depute[self.ptype] for depute in data[self.ptype_plural]]

    def parlementaire(self, slug_name):
        url = "%s/%s/%s" % (self.base_url, slug_name, self.format)
        return requests.get(url).json()[self.ptype]

    def picture(self, slug_name, pixels="60"):
        return requests.get(self.picture_url(slug_name, pixels=pixels))

    def picture_url(self, slug_name, pixels="60"):
        return "%s/%s/photo/%s/%s" % (self.base_url, self.ptype, slug_name, pixels)

    def search(self, q, page=1):
        # XXX : the response with json format is not a valid json :'(
        # Temporary return csv raw data
        url = "%s/recherche/%s?page=%s&format=%s" % (self.base_url, q, page, "csv")
        return requests.get(url).content

    @memoize
    def parlementaires(self, active=None):
        if active is None:
            url = "%s/%s/%s" % (self.base_url, self.ptype_plural, self.format)
        else:
            url = "%s/%s/enmandat/%s" % (self.base_url, self.ptype_plural, self.format)

        data = requests.get(url).json()
        return [depute[self.ptype] for depute in data[self.ptype_plural]]

    def search_parlementaires(self, q, field="nom", limit=5):
        return extractBests(
            q,
            self.parlementaires(),
            processor=lambda x: x[field] if type(x) == dict else x,
            limit=limit,
        )

    def interventions(self, dep_name, n_sessions=10, start=4850):
        name = self.search_parlementaires(dep_name)[0][0]["nom"]
        dep_intervention = []
        pattern = "(?<=Permalien" + name + ")" + ".*?(?=Voir tous les commentaires)"
        for num_txt in range(start, start + n_sessions):
            url = "https://www.nosdeputes.fr/15/seance/%s" % (str(num_txt))
            source = request.urlopen(url).read()
            # source.encoding = source.apparent_encoding
            page = bs4.BeautifulSoup(source, "lxml")
            x = re.findall(pattern, page.get_text(), flags=re.S)
            dep_intervention += x

        return dep_intervention

    def interventions2(self, dep_name):
        name = self.search_parlementaires(dep_name)[0][0]["nom"]
        name_pattern = re.sub(" ", "+", unidecode.unidecode(name.lower()))
        dep_intervention = []
        url = "https://www.nosdeputes.fr/recherche?object_name=Intervention&tag=parlementaire%3D{0}&sort=1".format(
            name_pattern
        )
        source = request.urlopen(url).read()
        page = bs4.BeautifulSoup(source, "lxml")
        for x in page.find_all("p", {"class": "content"}):
            dep_intervention += x

        return dep_intervention

    def liste_mots(self, dep_name):
        name = self.search_parlementaires(dep_name)[0][0]["nom"]
        name_pattern = re.sub(" ", "-", unidecode.unidecode(name.lower()))
        mots_dep = []
        url = "https://www.nosdeputes.fr/{0}/tags".format(name_pattern)
        source = request.urlopen(url).read()
        page = bs4.BeautifulSoup(source, "lxml")
        for x in page.find_all("span", {"class": "tag_level_4"}):
            mots_dep.append(re.sub("\n", "", x.get_text()))

        return mots_dep


# %% [markdown] id="zGU4ALalPCVS"
# Ensuite, nous avons créé plusieurs DataFrames à l'aide de la fonction interventions 2, avec les fonctions suivantes :

# %% id="eJlvLIZsPCVT"
# Fonctions intermédiaires


def deputies_of_group(group, n_deputies):
    all_names = deputies_df[deputies_df["groupe_sigle"] == group]["nom"]
    return all_names[:n_deputies]


def interventions_of_group(group, n_deputies=15):
    names = deputies_of_group(group, n_deputies)
    print(names)
    interventions = []
    for name in names:
        print(name)
        interventions += [[group, name, api.interventions2(name)]]
    return interventions


# Fonction de stockage des interventions


def stockintervention(groupe):
    interventions_group = []
    nbdep = deputies_df.groupby("groupe_sigle")["nom"].count()[str(groupe)]
    print(nbdep)
    interventions_group += interventions_of_group(groupe, nbdep)
    interventions_df = pd.DataFrame(
        interventions_group, columns=["groupe", "nom", "interventions"]
    )

    return interventions_df


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

# %% [markdown] id="EX1okKFfQcNn"
# ## Exploration et feature engineering

# %% id="tr7evwW0QdVy"
# Il faudrait faire un test-train-split ici, je crois

# %% id="1Yjhplu3QhBp"


# %% [markdown] id="p0ZAydeEPCVc"
# ## Visualisation des données (wordcloud, statistiques descriptives...)
#
# Nous allons maintenant dans cette partie visualiser les tendances dans les différents partis, ainsi que les mots qui sont les plus utilisés. Nous commençons par le wordcloud.

# %% id="AhEb0zObPCVd"
# Imports

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import matplotlib.pyplot as plt

# %matplotlib inline

from urllib import request

# Création d'un stopwords

stopping_list = request.urlopen(
    "https://raw.githubusercontent.com/rturquier/depythons/main/stopwords-fr.txt"
).read()
stopping_list = stopping_list.decode("utf-8")
stopwords_list = stopping_list.split("\n")


def wordcloud_gen(dep_name):
    name = api.search_parlementaires(dep_name)[0][0]["nom"]
    text_dep = api.interventions2(name)

    text_cloud = ""
    for morceau in text_dep:
        text_cloud += morceau

    stopwords = set(STOPWORDS)
    stopwords.update(stopwords_list)

    try_cloud = WordCloud(
        stopwords=stopwords, max_font_size=50, max_words=150, background_color="white"
    ).generate(text_cloud)

    plt.imshow(try_cloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(name)
    plt.show()


# %% [markdown] id="esjyOFUXPCVe"
# Quelques petits essais...

# %% id="OU4eFFXUPCVe" outputId="ab16e9fe-7214-45f6-e213-407f17829ccd"
wordcloud_gen("Jean-Luc Mélenchon"), wordcloud_gen("Eric Ciotti")

# %% [markdown] id="O-RmXt3xPCVf"
# ### Liste de mots customisée
#
# Créons dès à présent une fonction qui retourne les mots les plus utilisés par les membres d'un parti. Cette liste de mots va nous servir pour modéliser les champs lexicaux (quel parti a le plus tendance à utiliser tel mot ?).

# %% id="Xkllno7mPCVg"
from collections import Counter


def give_text(groupe_df):
    list_groupe = []
    for words in groupe_df["interventions"]:
        list_groupe.append(words)

    text_groupe = ""

    for block in list_groupe:
        for carac in block:
            text_groupe += carac

    return text_groupe


def customized(parti_df, nb_mots=100):
    parti_split = give_text(parti_df).split()
    parti_pure = [word for word in parti_split if word not in stopwords_list]
    parti_counter = Counter(parti_pure)
    parti_commons = parti_counter.most_common(nb_mots)

    customized_list = []
    for x in parti_commons:
        customized_list.append(x[0])

    return customized_list


# %% [markdown] id="Q1jlrJsUPCVh"
# Utilisons ces fonctions pour créer une liste de mots customisée, qui contient les mots les plus utilisés, partis de gauche et de droite confondus.

# %% id="I0lSH6JWPCVh" outputId="a243f78a-7908-4342-b754-89bfd768da17"
super_liste = customized(LFI_df) + customized(LR_df) + customized(SOC_df)
super_liste = list(set(super_liste))  # Suppression des doublons
super_liste

# %% id="xpEE2y5JPCVi"
