"""
Classe qui se charge de créer des bigram

Author : bsanchez@starclay.fr
date : 06/08/2020
"""

import gensim
import tempfile
import pickle
from datetime import datetime
import pandas as pd
import os
import s3fs
from .Process import Process
from .utils import *
import scipy
from gensim.models.phrases import Phrases, Phraser

class ProcessBigram(Process):
    
    def __init__(self,list_df_cols,path_run,id_run):
        """
        Classe permettant de transformer les inputs textuels en ngrams

        :param list_df_cols: liste des colonnes à traiter
        :param path_run: chemin de svg sur minio
        :param id_run: id du run de training
        """
        self.id_run = id_run
        self.path_run = path_run
        self.list_df_cols = list_df_cols
        self.list_model = []
        for col in self.list_df_cols:
            self.list_model.append(gensim.models.Phrases())

    def train_model(self,df_path):
        """
        Crée les dictionnaires de ngrams et les sauvegarde

        :param df_path: fichier à traiter
        """
        #Afin d'utiliser le moins de RAm possible on entraine 1 à 1 les modèles
        #pour les figer le plus vite
        for index,col in enumerate(self.list_df_cols):
            for df in pd.read_csv(df_path,chunksize=60000,dtype=str,sep=";",usecols=[str(col)]):
                    corpus_text = [str(x).split(" ") for x in df[col].values]
                    self.list_model[index].add_vocab(corpus_text)
            self.list_model[index] = Phraser(self.list_model[index])
        self.save_model(distant_file=True)

    def run_model(self,df_path):
        """
        transforme les données de :param df_path: via les modèles de ngrams entrainés

        :param df_path: fichier à traiter
        """
        new_name = incrementTmpFile(df_path)
                        
        for df in pd.read_csv(df_path,chunksize=60000,dtype=str,sep=";"):
            for index,col in enumerate(self.list_df_cols):
                # On a besoin d'une structure en bag of word pour gensim
                corpus_text = [str(x).split(" ") for x in df[col].values]
                corpus_text_w_bigram = [self.list_model[index][line] for line in corpus_text]
                # unflat la list pour qu'elle soit comme lors de la déclaration
                corpus_text_w_bigram = [" ".join(x) for x in corpus_text_w_bigram]
                df[col] = corpus_text_w_bigram
                       
            with open(F"tmp/{new_name}.csv", 'a') as f:
                df.to_csv(f, mode='a', sep=';', header=f.tell()==0, index=False)
                
        
    def apply(self,df_path):
        """
        extrait puis applique les modèles ngrams sur les données

        :param df_path: fichier à traiter
        """
        self.train_model(df_path)
        self.run_model(df_path)
        
    def save_model(self, distant_file=False, local_file=False):
        """
        Sauve les modèles localement et/ou sur minio
        """
        if local_file:
            for index,mod in enumerate(self.list_model):
                pickle.dump(mod, open(f"{self.id_run}_bigram_{list_df_cols[index]}", "wb" ))
        else:
            for index,mod in enumerate(self.list_model):
                f = tempfile.NamedTemporaryFile('w', delete=False)
                f.name = f"{self.id_run}_bigram_{self.list_df_cols[index]}"
                pickle.dump(mod, open(f.name, "wb"))
                save_file_on_minio(f.name, dir_name=self.path_run)
                os.remove(f.name)
                       
    # def load_model(distant_file=None, local_file=None):
    #     if not distant_file and not local_file:
    #         raise ValueError("At least 1 path must be given")
    #     if local_file:
    #         cl = CleanerRsX()
    #         cl.model = pickle.load(local_file)
    #         return cl
    #     else:
    #         f = tempfile.NamedTemporaryFile('w', delete=False)
    #         self.load_model_from_minio(distant_file, f.name)
    #         cl = CleanerRsX()
    #         cl.model = pickle.load(f.name)
    #         os.delete(f)
    #         return cl