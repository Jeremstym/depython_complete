# -*- coding: utf-8 -*-
"""
Created on Wed Dec  9 11:54:43 2020

Code de Rémi pour créer un DataFrame avec une intervention = une ligne

@author: Jérémie Stym-Popper
"""

import pandas as pd

data_url = "https://raw.githubusercontent.com/rturquier/depythons/main/Stock_csv/gd_inter.csv"
df_brut = pd.read_csv(data_url)
df_brut.sample(n=5)

# Création d'une indicatrice `droite` qui sera la cible de la classification
df_brut = df_brut.\
            assign(droite = df_brut["groupe"] == "LR").\
            sort_values(by = ["droite"], ascending = False)
df_brut.head()

# Équilibrage du nombre de députés
def balance_left_right(df):
    count = df.droite.value_counts()
    n_droite, n_gauche = count[True], count[False]
    df = df.sort_values(by=["droite"], ascending=False)
  
    if n_droite > n_gauche :
        df = df[n_droite - n_gauche:]
    elif n_droite < n_gauche :
        df = df[2 * n_droite:]
   
    return df

df_brut = balance_left_right(df_brut)
df_brut.droite.value_counts()

# Régler un problème de type
from ast import literal_eval
def convert_to_list(interventions):
    return literal_eval(str(interventions))

df_brut["interventions"] = df_brut["interventions"].apply(convert_to_list)


# Créer une feature "longeur de l'intervention"
df_brut = df_brut.assign(longeur = df_brut["interventions"].str.len())


