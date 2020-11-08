# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 22:33:57 2020
@author: Jérémie Stym-Popper (prise sur regardscitoyens)

Il s'agit du code permettant d'implémenter le 'cpc-api' manuellement. Cette API permet d'avoir accès à l'ensemble des députés et des sénateurs 
(ainsi que leur profil et leurs caractéristiques) sur les sites nosdeputes.fr et nossenateurs.fr
"""


# ----- Codage de l'API --------------------------------------------------
from operator import itemgetter
import requests
import warnings

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
    
    def interventions(self, dep_name):
        name = self.search_parlementaires(dep_name)[0][0]['nom']
        dep_intervention = []
        pattern = "(?<=Permalien)" + name + ".*Voir tous les commentaires"
        for num_txt in range(5000,5325):
            url = "https://www.nosdeputes.fr/15/seance/%s" % (str(num_txt))
            source = requests.get(url)
            source.encoding = source.apparent_encoding
            page = bs4.BeautifulSoup(source.text, "lxml")
            intervention = page.find_all("div", {"class":"intervention"})

            for parole in intervention:
                x = re.findall(pattern, parole.get_text(), flags=re.S)
                dep_intervention += x
        
        return dep_intervention


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
