# -*- coding: utf-8 -*-
"""
Created on Tue Dec  8 19:30:24 2020

Module pour créer une liste de mots customisée avant de modéliser

@author: Jérémie Stym-Popper
"""

import pandas as pd
import depute_api
from super_dataframe import LFI_df, LR_df, SOC_df

from collections import Counter

###--- Création d'un stopwords ----

stopping_list = open("stopwords-fr.txt", "r", encoding="utf8")
stopwords_list = stopping_list.read().split('\n')
stopping_list.close()

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
    

### ---- Des exemples d'utilisations----

customized(LFI_df)
