"""
Class pour communiquer avec le PostgreSQL

Author : bsanchez@starclay.fr
date : 16/07/2020
"""

import os
import s3fs
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import psycopg2
import yaml
import time
from tqdm import tqdm
import io


class PostGre_SQL_DB:
    
    def __init__(self, host, port, dbname, user, password, logger=None):
        """
        Classe de connexion à la base postgresql. Les informations nécessaires sont récupérées sur minio
        
        :param host, port, dbname, user, password: cf. __init__
        :param logger: logger à utiliser
        """
        self.HOST       = host
        self.PORT       = port
        self.DBNAME     = dbname
        self.USER       = user
        self.PASSWORD   = password
        self.logger = logger
        # un engine est un dialect, il ne se connecte pas à la bdd
        self.engine = create_engine(F'postgresql://{self.USER}:{self.PASSWORD}@{self.HOST}:{self.PORT}/{self.DBNAME}',
                                    client_encoding='utf8', encoding='utf8', convert_unicode=True)

    def read_from_table(self,table_name):
        '''
        Renvoie une DataFrame Pandas à partir d'une table entière (non recommandé)

        :param table_name: nom de la table à récupérer
        :return: pd.DataFrame
        '''
        df = pd.read_sql_table(F'{table_name}', con=self.engine)
        return df
    
    def read_from_sql(self,sql_query,chunksize=0):
        '''
        Renvoie une DataFrame Pandas avec une ligne = 1 enregistrement correspondant à sql_query

        :param sql_query: str, query sql à appliquer
        :param chunksize: taille de chunk à appliquer (permet de ne charger les données que bout par bout)
        :return: pd.DataFrame ou itérateur de pd.DataFrame
        '''
        if chunksize > 0:
            df = self.read_from_sql_with_chunksize(sql_query, chunksize)
        else:
            df = pd.read_sql(F'{sql_query}', con=self.engine)
        return df

    def read_from_sql_with_chunksize(self,sql_query,chunksize):
        """
        Renvoie une DataFrame Pandas avec une ligne = 1 enregistrement correspondant à sql_query
        Ici on passe sur psycopg2 car c'est le seul driver python pgsql qui gère le chunksize correctement

        (les autres chargent toutes les données en mémoire puis les découpent en dataframes)
        
        :param sql_query: str, query sql à appliquer
        :param chunksize: taille de chunk à appliquer (permet de ne charger les données que bout par bout)
        :return: pd.DataFrame ou itérateur de pd.DataFrame
        """
        with psycopg2.connect("host='{}' port={} dbname='{}' user={} password={}".format(self.HOST, self.PORT, self.DBNAME, self.USER, self.PASSWORD)) as conn:
            curPG = conn.cursor('testCursor', cursor_factory=psycopg2.extras.DictCursor)
            curPG.itersize = chunksize
            curPG.execute(sql_query)
            l = []
            for rec in curPG:
                l.append(rec.copy())
                if len(l) == chunksize:
                    yield pd.DataFrame(l)
                    l = []
            if l:
                yield pd.DataFrame(l)
    
    def _log(self, msg):
        if self.logger:
            self.logger.info(msg)