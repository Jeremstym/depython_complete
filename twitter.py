import pandas as pd
import json
import numpy as np
import snscrape.modules.twitter as sntwitter
import io
import depute_api


def get_tweet(idtwitter, nb):
    # Prends un utilisateur twitter et renvoie ses nb derniers tweets

    tweets_list = []
    # TwitterSearchScraper
    for i,tweet in enumerate(sntwitter.TwitterSearchScraper('from:'+idtwitter).get_items()):
        if i>nb:
            break
        tweets_list.append([tweet.date, tweet.id, tweet.content, tweet.username])

    # Dataframe crée avec les catégories suivantes
    tweets_df = pd.DataFrame(tweets_list, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])
    return tweets_df

def list_txt(df):
    # Renvoie une liste des tweets de la df en question
    list_txt = []
    for txt in df['Text']:
        list_txt.append(txt)
    return list_txt


api = depute_api.CPCApi()
deputies_json = api.parlementaires()
deputies_df = pd.json_normalize(deputies_json)

twitter_df = deputies_df[["nom","nom_de_famille","sexe","date_naissance","groupe_sigle","twitter"]].drop(deputies_df[deputies_df.twitter == ""].index)

def df_twitterbuild(df):
    # Crée une dataframe des députés avec les 500 derniers tweets de chaque députés
    twitter_df = df[["nom","nom_de_famille","sexe","date_naissance","groupe_sigle","twitter"]].drop(df[df.twitter == ""].index)
    twitter_df["Tweets"] = ""
    for idtwitter in twitter_df.twitter:
        i = twitter_df.loc[twitter_df['twitter'] == idtwitter].index.values[0]
        twitter_df.loc[i, "Tweets"] = list_txt(get_tweet(idtwitter,500))
    return twitter_df

twitter_df = df_twitterbuild(deputies_df)

twitter_df.to_csv(r"C:/Users/Luca/Documents/GitHub/depythons/Twitter.csv")
twitter_df.to_json(r"C:/Users/Luca/Documents/GitHub/depythons/Twitter.json")
