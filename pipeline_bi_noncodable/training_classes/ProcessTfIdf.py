"""
Classe qui créer une matrice tf-idf

Author : bsanchez@starclay.fr
date : 06/08/2020
"""

import tempfile
import pickle
from datetime import datetime
from itertools import zip_longest

import pandas as pd
from tqdm import tqdm
import os
from .utils import *
from scipy import sparse
import csv

import numpy as np
from numpy import array

import gensim
import gensim.downloader as api
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
from gensim import corpora, matutils, models, similarities
from gensim.matutils import corpus2dense, corpus2csc

import scipy.sparse
from scipy.sparse import hstack

from .Process import Process

def top_n_words(matrix,nword):
    """
    Retrouve les n mots les plus fréquent dans une matrice.
    
    :params matrix: Matrice creuse de type scipy
    :params nword: Integer, nb de mots que l'on souhaite
    
    :returns: words: list des n mots les plus fréquent 
    
    """
    count = {}
    row, col = matrix.nonzero()
    for row, col in tqdm(zip(row, col)):
        if row in count:
            count[row] = count[row] + 1
        else:
            count[row] = 1
    words = list(count.items())
    words.sort(key=lambda tup: tup[1],reverse=True)
    words = words[0:nword]
    words = [i[0] for i in words]
    return words

def file_len(fname):
    with open(fname) as f:
        for i, l in enumerate(f):
            pass
    return i + 1

class ProcessTfIdf(Process):
    
    def __init__(self,list_df_cols,path_run,id_run,len_tfidf=1000):
        """
        Classe permettant de transformer les inputs textuels en tfidf

        :param list_df_cols: liste des colonnes à traiter
        :param path_run: chemin de svg sur minio
        :param id_run: id du run de training
        :param len_tfidf: int, taille du tfidf (nb de mots à garder)
        """
        self.id_run = id_run
        self.path_run = path_run
        self.list_df_cols = list_df_cols
        self.list_dict = []
        self.corpus = []
        self.len_tfidf = 1000
        for col in self.list_df_cols:
            self.list_dict.append(Dictionary())
            self.corpus.append([])

    def train_model(self,df_path):
        """
        Entraine le modèle en transformant le text en BoW. 

        :param df_path: fichier à traiter
        """
        for df in pd.read_csv(df_path,chunksize=60000,dtype=str,sep=";"):
            for index,col in enumerate(self.list_df_cols):
                corpus_text = [str(x).split(" ") for x in df[col].values]
                corp = []
                for line in corpus_text:
                    self.list_dict[index].add_documents([line])
                    self.corpus[index].append(self.list_dict[index].doc2bow(line))
                    

    def run_model(self,df_path):
        """
        transforme les données de :param df_path: via les modèles de tfidf entrainés
        
        :param df_path: fichier à traiter
        """
        nb_rows = file_len(df_path)
        
        for index_col,col in enumerate(self.list_df_cols): 
            
            new_name = incrementTmpFile(F"{df_path}")
            
            tf_cols_name =  {v: k for k, v in self.list_dict[index_col].token2id.items()}
            
            for key in tf_cols_name.keys():
                tf_cols_name[key] = f'{col}_'.join(tf_cols_name[key])
                        
            tfidf_model = TfidfModel(self.corpus[index_col])
            
            tf_sparse_array = matutils.corpus2csc(self.corpus[index_col])
            
            tfidf_matrix = tfidf_model[self.corpus[index_col]]

            tops = top_n_words(tf_sparse_array,self.len_tfidf)
            
            f = open(f"vocab_tfidf_{self.id_run}.txt","w")
            f.write( str(tf_cols_name) )
            f.close()
            save_file_on_minio(f"vocab_tfidf_{self.id_run}.txt", self.path_run)
            os.remove(f"vocab_tfidf_{self.id_run}.txt")
            
            f = open(f"top_tfidf_{self.id_run}.txt","w")
            f.write( str(tops) )
            f.close()
            save_file_on_minio(f"top_tfidf_{self.id_run}.txt", self.path_run)
            os.remove(f"top_tfidf_{self.id_run}.txt")
            
            smatrix = corpus2csc(tfidf_matrix,num_terms=len(tf_cols_name),num_docs=nb_rows)
            smatrix = smatrix.T
            
            smatrix = smatrix[:,tops]
            
            scipy.sparse.save_npz(F"tmp/tfidf_{col}.npz",smatrix)
        
        if len(self.list_df_cols) > 1:
            matrix = []
            for col in self.list_df_cols:
                m = scipy.sparse.load_npz(F"tmp/tfidf_{col}.npz")
                matrix.append(m)
                
            sp_full = hstack(matrix)
            scipy.sparse.save_npz(F"tmp/{new_name}.npz",sp_full)
        else:
            os.rename(f'tmp/tfidf_{self.list_df_cols[0]}.npz',f'tmp/{new_name}.npz')
            
    def apply(self,df_path):
        """
        extrait puis applique les modèles ngrams sur les données

        :param df_path: fichier à traiter
        """
        self.train_model(df_path)
        self.run_model(df_path)
        
    # def save_model(self, distant_file=None, local_file=None):
    #     if not distant_file and not local_file:
    #         raise ValueError("At least 1 path must be given")
    #     if local_file:
    #         pickle.dump(self.model, open( local_file, "wb" ) )
    #     if not f:
    #         f = tempfile.NamedTemporaryFile('w', delete=False)
    #         pickle.dump(self.model, open( f, "wb" ) )
    #         self.save_model_on_minio(f.name, distant_file)
    #         os.delete(f)

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