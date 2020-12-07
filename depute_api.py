# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 22:33:57 2020

@author: Jérémie Stym-Popper
"""


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


__all__ = ['CPCApi']


def memoize(f):
    cache = {}

    def aux(*args, **kargs):
        k = (args, tuple(sorted(kargs.items())))
        if k not in cache:
            cache[k] = f(*args, **kargs)
        return cache[k]
    return aux


class CPCApi(object):
    format = 'json'

    def __init__(self, ptype='depute', legislature=None):
        """
        type: depute or senateur
        legislature: 2007-2012 or None
        """

        assert(ptype in ['depute', 'senateur'])
        assert(legislature in ['2007-2012', '2012-2017', None])
        self.legislature = legislature
        self.ptype = ptype
        self.ptype_plural = ptype + 's'
        self.base_url = 'https://%s.nos%s.fr' % (legislature or 'www', self.ptype_plural)

    def synthese(self, month=None):
        """
        month format: YYYYMM
        """
        if month is None and self.legislature == '2012-2017':
            raise AssertionError('Global Synthesis on legislature does not work, see https://github.com/regardscitoyens/nosdeputes.fr/issues/69')

        if month is None:
            month = 'data'

        url = '%s/synthese/%s/%s' % (self.base_url, month, self.format)

        data = requests.get(url).json()
        return [depute[self.ptype] for depute in data[self.ptype_plural]]

    def parlementaire(self, slug_name):
        url = '%s/%s/%s' % (self.base_url, slug_name, self.format)
        return requests.get(url).json()[self.ptype]

    def picture(self, slug_name, pixels='60'):
        return requests.get(self.picture_url(slug_name, pixels=pixels))

    def picture_url(self, slug_name, pixels='60'):
        return '%s/%s/photo/%s/%s' % (self.base_url, self.ptype, slug_name, pixels)

    def search(self, q, page=1):
        # XXX : the response with json format is not a valid json :'(
        # Temporary return csv raw data
        url = '%s/recherche/%s?page=%s&format=%s' % (self.base_url, q, page, 'csv')
        return requests.get(url).content
    
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
        name_pattern = re.sub(' ', '+', unidecode.unidecode(name.lower()))
        dep_intervention = []
        url = "https://www.nosdeputes.fr/recherche?object_name=Intervention&tag=parlementaire%3D{0}&sort=1".format(name_pattern)
        source = request.urlopen(url).read()            
        page = bs4.BeautifulSoup(source, "lxml")
        for x in page.find_all('p', {'class' : 'content'}):
            dep_intervention += x

        return dep_intervention
    
    def liste_mots(self, dep_name):
        name = self.search_parlementaires(dep_name)[0][0]["nom"]
        name_pattern = re.sub(' ', '-', unidecode.unidecode(name.lower()))
        mots_dep = []
        url = "https://www.nosdeputes.fr/{0}/tags".format(name_pattern)
        source = request.urlopen(url).read()            
        page = bs4.BeautifulSoup(source, "lxml")
        for x in page.find_all('span', {'class' : 'tag_level_4'}):
            mots_dep.append(re.sub("\n", "", x.get_text()))
            
        return mots_dep
    
    
    @memoize
    def parlementaires(self, active=None):
        if active is None:
            url = '%s/%s/%s' % (self.base_url, self.ptype_plural, self.format)
        else:
            url = '%s/%s/enmandat/%s' % (self.base_url, self.ptype_plural, self.format)

        data = requests.get(url).json()
        return [depute[self.ptype] for depute in data[self.ptype_plural]]
    
    def search_parlementaires(self, q, field='nom', limit=5):
        return extractBests(q, self.parlementaires(), processor=lambda x: x[field] if type(x) == dict else x, limit=limit)


#---------- Importer la table ------------------
        
import pandas as pd

import depute_api


api = depute_api.CPCApi()

deputies_json = api.parlementaires()
deputies_df = pd.json_normalize(deputies_json)


deputies_df.head()


#------ C'est la solution ! ---------------
#l = []            
#for parler in intervient:
#    l += re.findall('(?<=Permalien)Jean Castex.*Voir tous les commentaires',
#                    parler.get_text(), flags=re.S)


# Bonjour je fais un test de commentaire pour faire du git
            
        

