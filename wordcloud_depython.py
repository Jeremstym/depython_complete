# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 15:41:22 2020

Ce module consiste en la création d'un word cloud, fondée sur du NLP 
pour le projet "Dépython".

@author: Jérémie Stym-Popper
"""

import numpy as np
import pandas as pd
from depute_api import CPCApi
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt

#------ Implémentation de la class CPCApi------------

api = CPCApi()

text_dep = api.interventions2('Jean-Luc Mélenchon')

#---- Création d'un stop words ------

stopping_list = open("stopwords-fr.txt","r", encoding="utf8")
stopwords_list = stopping_list.read().split('\n')
stopping_list.close()

#----------Création du word cloud ----------

text_cloud = ""
for morceau in text_dep:
    text_cloud += morceau
    
stopwords = set(STOPWORDS)
stopwords.update(stopwords_list)

try_cloud = WordCloud(stopwords = stopwords,
                      max_font_size=50, max_words=150, 
                      background_color="white").generate(text_cloud)

plt.imshow(try_cloud, interpolation='bilinear')
plt.axis("off")
plt.show()


#----- Fonction génératrice d'un wordcloud-----

# Ne pas oublier de poser api = CPCApi() 

def wordcloud_gen(dep_name):
    name = api.search_parlementaires(dep_name)[0][0]['nom']
    text_dep = api.interventions2(name)
    
    text_cloud = ""
    for morceau in text_dep:
        text_cloud += morceau
    
    stopwords = set(STOPWORDS)  
    stopwords.update(stopwords_list)

    try_cloud = WordCloud(stopwords = stopwords,
                          max_font_size=50, max_words=150, 
                          background_color="white").generate(text_cloud)

    plt.imshow(try_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show()
    