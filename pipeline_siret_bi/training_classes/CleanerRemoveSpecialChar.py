"""
Classe de nettoyage des données

Author : bsanchez@starclay.fr
date : 06/08/2020
"""
import os
import preprocessing as pre
import pandas as pd

import nltk 
nltk.download('stopwords')
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer


class CleanerRemoveSpecialChar:
    """
    Cleaner qui nettoie les données en entrée.
    Supression des caractère spéciaux
    Suprresion des stop word
    Stemming des mots
    """
    def __init__(self, list_df_cols):
        """
        Cleaner qui nettoie les données en entrée.
        Supression des caractère spéciaux
        Suprresion des stop word
        Stemming des mots

        :param list_df_cols: liste des noms de colonne à traiter
        """
        self.list_df_cols = list_df_cols
        
    def process(self, input_file, output_file):
        """
        Applique le traitement 
        
        :param path: chemin du csv
        """
        for df in pd.read_csv(input_file, chunksize=60000, dtype=str, sep=";"):
        
            for col in self.list_df_cols:
                df[col].fillna(f'{str(col).replace("_","")}inconnue',inplace=True)

            nltk_stopwords = nltk.corpus.stopwords.words('french')
            stemmer = SnowballStemmer(language='french')

            for col in self.list_df_cols:
                df[col] = df[col].apply(lambda x: pre.clean(x))
                df[col] = df[col].apply(lambda x: ' '.join([word for word in x.split() if word not in (nltk_stopwords)]))
                df[col] = df[col].apply(lambda x: " ".join([stemmer.stem(word) for word in x.split(" ")]))

            with open(output_file, 'a') as f:
                df.to_csv(f, mode='a', sep=';', header=f.tell()==0,index=False)