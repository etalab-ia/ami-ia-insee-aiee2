"""
Classe qui se charge de créer des vector document

https://radimrehurek.com/gensim/auto_examples/tutorials/run_doc2vec_lee.html#sphx-glr-auto-examples-tutorials-run-doc2vec-lee-py

Author : bsanchez@starclay.fr
date : 06/08/2020
"""

import gensim
import tempfile
import pickle
from datetime import datetime
import pandas as pd
from datetime import datetime
import os
import s3fs
from .Process import Process
from .utils import *
import scipy
from gensim.models.phrases import Phrases, Phraser

from tensorflow.keras.preprocessing.text import Tokenizer
from gensim.models.fasttext import FastText
import numpy as np
import nltk
from string import punctuation
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk import WordPunctTokenizer
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import time


class ProcessDoc2Vec(Process):
    
    def __init__(self,list_df_cols,path_run,id_run,dict_models={}):
        """
        Classe permettant de transformer les inputs textuels en doc2vec

        :param list_df_cols: liste des colonnes à traiter
        :param path_run: chemin de svg sur minio
        :param id_run: id du run de training
        :param dict_models: dictionnaires modèles (list of dict)
        """
        self.name = "Doc2Vec"
        self.id_run = id_run
        self.path_run = path_run
        self.list_df_cols = list_df_cols
        self.list_model = []
        self.dict_models = dict_models
        for col in self.list_df_cols:
            self.list_model.append(None)

    
    def train_model(self,df_path):
        """
        Crée un modèle Doc2Vec par colonne

        :param df_path: fichier à traiter
        """
        
        start = time.time()
        
        flatten = lambda l: [item for sublist in l for item in sublist]
        
        for index,col in enumerate(self.list_df_cols):
            if col in self.dict_models[col]:
                with open(self.dict_models[col], 'rb') as handle:
                    self.list_model[index] = pickle.load(handle)
            else:
                train_corpus = []
                for df in pd.read_csv(df_path,chunksize=60000,dtype=str,sep=";",usecols=[str(col)]):
                        train_corpus.append(df[col].tolist())

                #On transforme le corpus en document 
                train_corpus = flatten(train_corpus)
                train_corpus = [TaggedDocument(words=word_tokenize(str(_d).lower()), tags=[str(i)]) for i, _d in enumerate(train_corpus)]

                #Hyperparametre, comment les choisir ?
                epochs = 60
                vec_size = 30
                 # normalement on utilise cpu_count mais vu qu'il s'agit d'un environnement en contenaire
                 # la fonction retour tout les cores du clusteurs (~72)
                cores = 9


                model = gensim.models.doc2vec.Doc2Vec(vector_size=vec_size, min_count=2, epochs=epochs)
                model.build_vocab(train_corpus)
                model.train(train_corpus, total_examples=model.corpus_count, epochs=model.epochs)

                self.list_model[index] = model
                phaseC = time.time()
                print(f"Doc2Vec train complete {col} {phaseC - start}")

                # Sanity check: On calcule le vector pour chaque document du training
                # Si le modèle est sain, le document le plus similaire est lui même.
                # La doc officielle est fausse => TODO
                
        phase1 = time.time()
        print(f"Doc2Vec train complete: {phase1 - start}")
            
        self.save_model(distant_file=True)

    def run_model(self,df_path):
        """
        transforme les données de :param df_path: via les modèles de Doc2Vec entrainés

        :param df_path: fichier à traiter
        """
        
        start = time.time()
        
        flatten = lambda l: [item for sublist in l for item in sublist]
        
        new_name = incrementTmpFile(df_path)
                        
        for df in pd.read_csv(df_path,chunksize=60000,dtype=str,sep=";"):
            list_document_vector = []
            for index_row,row in df.iterrows():
                document_vector = []
                for index,col in enumerate(self.list_df_cols):
                    col_tokens = str(row[col]).split(" ")
                    col_vector = self.list_model[index].infer_vector(col_tokens)
                    document_vector.append(col_vector)
                document_vector = flatten(document_vector)
                list_document_vector.append(document_vector)
            df_document_vector = pd.DataFrame(list_document_vector)
            with open(F"tmp/{new_name}.csv", 'a') as f:
                df_document_vector.to_csv(f, mode='a', sep=';', header=f.tell()==0, index=False)
                
        phase1 = time.time()
        print(f"Time consumed in working: {phase1 - start}")
                
        
    def apply(self,df_path):
        """
        extrait puis applique les modèles Doc2Vec sur les données

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
                pickle.dump(mod, open(f"{self.id_run}_d2v_{list_df_cols[index]}", "wb" ))
        else:
            for index,mod in enumerate(self.list_model):
                f = tempfile.NamedTemporaryFile('w', delete=False)
                f.name = f"{self.id_run}_{self.name}_{self.list_df_cols[index]}"
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