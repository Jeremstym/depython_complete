# -*- coding: utf-8 -*-
"""
Created on Fri Nov 20 13:50:25 2020

Le but de ce module est de développer un moyen de transformer les 
chaînes de caractère en vecteurs ou matrices, afin d'effectuer des 
rapprochements et détecter des caractéristiques propres aux textes.

@author: Jérémie Stym-Popper
"""

import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.feature_extraction.text import CountVectorizer
import re
import depute_api 
import numpy as np

path_csv = r"C:\Users\Asus\Desktop\Jérémie\Fac_ENSAE\Informatique\Datapython_2AS1\Projet\interventions.csv"
df_inter = pd.read_csv(path_csv)


#-------- Création d'une liste de vocabulaire pour comparer les interventions-

"""
Le but est ici de créer une liste de mots sur lesquels seront comparées 
les interventions des députés. Cette liste permet de créer un même 
élément de comparaison pour rendre les calculs plus performants.
"""


text_gen = []
for i in df_inter['interventions']:
    text_gen.append(i)

transformer_all = CountVectorizer()
transformer_all.fit_transform(text_gen)

voc = transformer_all.vocabulary_.keys()

"""
transfomer_all2 = CountVectorizer(vocabulary=voc)  
X_all = transfomer_all2.fit_transform(text_gen)
"""

#------ Création de CountVectorizer pour les groupes parlementaires -------
"""
text_LREM=[]
for i in df_inter[df_inter['groupe']=='LREM']['interventions']:
    text_LREM.append(i)

len(text_LREM)

transformer_LREM = CountVectorizer(vocabulary=voc)

X_LREM = transformer_LREM.fit_transform(text_LREM)
X_LREM.toarray()



text_LFI=[]
for i in df_inter[df_inter['groupe']=='LFI']['interventions']:
    text_LFI.append(i)

len(text_LFI)

transformer_LFI = CountVectorizer(vocabulary=voc)

X_LFI = transformer_LFI.fit_transform(text_LFI)
X_LFI.toarray()
"""

def countervect(df_parole, groupe=None):
    if groupe != None:
        text_list = []
        for i in df_parole[df_parole['groupe']=='{0}'.format(str(groupe))]['interventions']:
            text_list.append(i)
        transformer_group = CountVectorizer(vocabulary=voc)
        X_groupe = transformer_group.fit_transform(text_list)
        
    else:
        text_gen = []
        for i in df_parole['interventions']:
            text_gen.append(i)
        
        transfomer_count = CountVectorizer(vocabulary=voc)  
        X_groupe = transfomer_count.fit_transform(text_gen)
        
    return X_groupe.toarray()
    
#---- Création d'un df contenant toutes les interventions par député ----

"""
api = depute_api.CPCApi()
deputies_json = api.parlementaires()
deputies_df = pd.json_normalize(deputies_json)

serie_countvect = pd.Series(
    list(X_all.toarray()),
    index=list(df_inter['groupe'])
    )

df_countervect = pd.DataFrame(serie_countvect)
print(df_countervect)
"""

def counter_maker(df_parole):
    matrix = np.matrix(countervect(df_parole))
    df_counter = pd.DataFrame(matrix, index=df_parole['groupe'])
    return df_counter
