# Code de visualisation en fréquence des tweets des députés

import collections
£import re
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import spacy


df_twitter = pd.read_csv((
    "https://raw.githubusercontent.com/rturquier/depythons/main/Stock_csv/Twitter.csv"
))

df_brut = pd.read_csv((
    "https://raw.githubusercontent.com/rturquier/depythons/main/Stock_csv/gd2_inter.csv"
))

df_brut
sp = spacy.load("fr_core_news_sm")
stop_words = sp.Defaults.stop_words | {"'", ",", ";", ":", " ", "", ".", ""}

df_twitter = pd.merge(df_twitter, df_brut, how = 'inner', on "nom")[["nom","droite","Tweets"]]
df_twitter = df_twitter.assign(droite=df_twitter["groupe_sigle"] == "LR")[["nom","droite","Tweets"]]
df_twitter = df_twitter.sort_values(["droite"])

df_twitter["Tweets"]


wordcount_droite = collections.defaultdict(int)
wordcount_gauche = collections.defaultdict(int)
df_twitter_droite = df_twitter[df_twitter["droite"] == True]
df_twitter_gauche = df_twitter[df_twitter["droite"] != True]
for inters in df_twitter_droite["Tweets"]:
    for word in inters:
        word = re.sub(r"\W", "", word)
        if word not in stop_words:
            wordcount_droite[word] += 1
for inters in df_twitter_gauche["Tweets"]:
    print(inters)
