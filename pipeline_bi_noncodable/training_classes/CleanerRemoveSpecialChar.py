"""
Classe de nettoyage des données

Author : bsanchez@starclay.fr
date : 06/08/2020
"""

import os
import gensim
import tempfile
import pickle
from datetime import datetime
import preprocessing as pre
import pandas as pd
import nltk 
nltk.download('stopwords')



class CleanerRemoveSpecialChar:

    def __init__(self,list_df_cols):
        """
        Classe de nettoyage des données
        Nettoie les champs, et créée la target du training si besoin

        :param list_df_cols: liste des noms de colonnes à traiter et garder
        """
        self.list_df_cols = list_df_cols
        
    def process(self,input_file, output_dir='tmp'):
        """
        :param input_file: fichier à traiter (pd.Dataframe en csv)
        :param output_dir: où écrire le résultats (dans output_dir/1.csv)
        :return: None
        """
        for df in pd.read_csv(input_file,chunksize=60000,dtype=str,sep=";"):
            # cols_to_keep = ['cabbi','rs_x',
            #                 'ilt_x', 'vardompart_x',
            #                 'numvoi_x', 'bister_x',
            #                 'typevoi_x','nomvoi_x',
            #                 'cpladr_x', 'actet_x',
            #                 'clt_x', 'dlt_x',
            #                 'plt_x', 'profi_x',
            #                 'profs_x', 'profa_x',
            #                 'siretm','siretc']
            # df = df[cols_to_keep]
            df = df[self.list_df_cols]

            ####################################    
            # Création de la colonne Target
            ####################################
            if 'siretm' in df.columns:
                df['siretm'].fillna("",inplace=True)
                df['siretm'] = df['siretm'].apply(lambda x: x.strip())
            if 'siretc' in df.columns:
                df['siretc'].fillna("",inplace=True)
                df['siretc'] = df['siretc'].apply(lambda x: x.strip())

            df['target'] = 0
            if 'siretm' in df.columns:
                df['target'] = df.apply(lambda x: 1 if x['siretm'] == "" and x['siretc'] == "" else 0,axis=1)
            ###############################
            # 
            ###############################
        
            for col in self.list_df_cols:
                df[col].fillna(f'{str(col).replace("_","")}inconnue',inplace=True)

            nltk_stopwords = nltk.corpus.stopwords.words('french')
            
            for col in self.list_df_cols:
                df[col] = df[col].apply(lambda x: pre.clean(x))
                df[col] = df[col].apply(lambda x: ' '.join([word for word in x.split() if word not in (nltk_stopwords)]))
                

            with open(os.path.join(output_dir, "1.csv"), 'a') as f:
                df.to_csv(f, mode='a', sep=';', header=f.tell()==0,index=False)