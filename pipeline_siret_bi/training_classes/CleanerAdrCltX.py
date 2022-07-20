"""
Classe de nettoyage des données

Author : bsanchez@starclay.fr
date : 06/08/2020
"""
import pandas as pd


class CleanerAdrCltX:
    """
    Cleaner qui nettoie la colonne adr_et_l6 de la base sirus afin qu'elle soit au même format que la colonne clt_x de la base BI
    """
    def __init__(self, list_df_cols=None):
        """
        Cleaner qui nettoie la colonne adr_et_l6 de la base sirus afin qu'elle soit au même format que la colonne clt_x de la base BI

        :param list_df_cols: non utilisé
        """
        self.list_df_cols = list_df_cols
        
    def process(self, input_file, output_file):
        """
        Applique le traitement 
        
        :param path: chemin du csv
        """
        for df in pd.read_csv(input_file, chunksize=60000, dtype=str, sep=";"):
            df['adr_et_l6'] = df['adr_et_l6'].str[5:]
            
            with open(output_file, 'a') as f:
                df.to_csv(f, mode='a', sep=';', header=f.tell()==0,index=False)