"""
Classe de nettoyage des données

Author : bsanchez@starclay.fr
date : 06/08/2020
"""
import pandas as pd


class CleanerActetX:
    """
    Cleaner qui nettoie la colonne actet_x
    """
    def __init__(self, list_df_cols=None):
        """
        Cleaner qui nettoie la colonne actect_x

        :param list_df_cols: non utilisé
        """
        self.list_df_cols = list_df_cols
        
    def process(self, input_file, output_file):
        """
        Applique le traitement 
        
        :param path: chemin du csv
        """
        for df in pd.read_csv(input_file, chunksize=60000, dtype=str, sep=";"):
            with open(output_file, 'a') as f:
                df.to_csv(f, mode='a', sep=';', header=f.tell()==0,index=False)