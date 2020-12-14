# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 19:30:24 2020

Module pour créer une liste de mots customisée avant de modéliser

@author: Jérémie Stym-Popper
"""

import pandas as pd
import depute_api
from urllib import request

from collections import Counter

# Création d'un stopwords

stopping_list = request.urlopen(
    "https://raw.githubusercontent.com/rturquier/depythons/main/stopwords-fr.txt"
).read()
stopping_list = stopping_list.decode("utf-8")
stopwords_list = stopping_list.split("\n")

###---- Création de la liste de mots customisée --------

def give_text(groupe_df):
    list_groupe = []
    for words in groupe_df['interventions']:
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

###---- Création de la liste de mots pour la modélisation ----

LFI_df = pd.read_csv("https://raw.githubusercontent.com/rturquier/depythons/main/Stock_csv/LFI2_inter.csv")
LR_df = pd.read_csv("https://raw.githubusercontent.com/rturquier/depythons/main/Stock_csv/LR2_inter.csv")
SOC_df = pd.read_csv("https://raw.githubusercontent.com/rturquier/depythons/main/Stock_csv/SOC2_inter.csv")

super_liste = customized(LFI_df) + customized(LR_df) + customized(SOC_df)

super_liste = list(set(super_liste)) # Suppression des doublons
