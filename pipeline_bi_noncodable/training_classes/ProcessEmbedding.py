"""
Classe qui se charge de créer des embedding

Author : bsanchez@starclay.fr
date : 06/08/2020
"""
from tqdm import tqdm

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
from gensim.corpora import Dictionary

#keras
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM
from tensorflow.keras.models import Model

from nltk.probability import FreqDist

from numpy import asarray
from numpy import save
from numpy import load

# https://github.com/keras-team/keras/issues/13118
import tensorflow.keras.backend as K


class ProcessEmbedding(Process):
    
    def __init__(self, list_df_cols, path_run, id_run, dict_models={}):
        """
        Classe permettant de transformer les inputs textuels en vecteurs
        via un modèle d'embeddings entrainé par la classe

        :param list_df_cols: liste des colonnes à traiter
        :param path_run: chemin de svg sur minio
        :param id_run: id du run de training
        :param dict_models: dictionnaires modèles (list of dict)
        """
        self.name = "Embedding"
        self.id_run = id_run
        self.path_run = path_run
        self.list_df_cols = list_df_cols
        self.dico_vocab = []
        self.list_model = []
        self.list_len_largest_token = []
        self.encoded_corpus = []
        for col in self.list_df_cols:
            self.list_model.append(None)
            self.dico_vocab.append(Dictionary())
            self.encoded_corpus.append([])
            self.list_len_largest_token.append(-1)
            
            
        self.dict_models = {}

    def train_model(self,df_path):
        """
        On crée les modeles d'embedding. 
        Transformation d'une liste de type ["groupe pomona","carrefour"]
        en list de list de token [["1","2"],["3","0"]] avec padding où 0 réprésente l'absence d'un mot.
        Le padding est calqué sur la plus grande list de token.
        La taille du vocabulaire est calculé et transmis au modèle.
        
        Retourne un fichier npy contenant tout les embedding

        :param df_path: fichier à traiter
        """
        start = time.time()
        
        flatten = lambda l: [item for sublist in l for item in sublist]
            
        for index, col in enumerate(self.list_df_cols):
            if 1 == 2: #BUG if col in dict_models:
                with open(self.dict_models[col], 'rb') as handle:
                    self.list_model[index] = pickle.load(handle)
            else:
                dict_len_col = {
                    "rs_x": 256
                    ,"clt_x": 256
                    ,"profs_x":256
                    ,"profi_x":128
                    ,"profa_x":128
                    ,"numvoi_x":2
                    ,"typevoi_x":2
                    ,"actet_x":256
                    ,"dlt_x":256
                    ,"plt_x":2
                    ,"vardompart_x":2
                }
                
                LEN_VECTOR = dict_len_col[col]
                
                df = pd.read_csv(df_path, dtype=str, sep=";", usecols=[str(col)])
                train_corpus = df[col].values.tolist()
                
                nb_docs = len(train_corpus)
                
                self.list_len_largest_token[index] = -1
                
                fdist = FreqDist()
                
                #Determiner la plus grande list de token 
                for index_doc, doc in enumerate(train_corpus):
                    tokens = str(doc).split(" ")
                    self.dico_vocab[index].add_documents([tokens])
                    for token in tokens:
                        fdist[token.lower()] += 1
                    if(self.list_len_largest_token[index] < len(tokens)):
                        self.list_len_largest_token[index] = len(tokens)
                        
                more_than_one = list(filter(lambda x: x[1] >= 2, fdist.items()))
                
                vocab_size = len(self.dico_vocab[index])
                
                print(f"vocab_size {vocab_size} (filtré) vs {len(self.dico_vocab[index])}")
                
                print(f"largest token is: {self.list_len_largest_token[index]}")
                
                for x in train_corpus:
                    self.encoded_corpus[index].append(one_hot(str(x),vocab_size))
                    
                del train_corpus
                for x in range(len(self.encoded_corpus[index])):
                    self.encoded_corpus[index][x] = flatten(pad_sequences([self.encoded_corpus[index][x]]
                                                                          , maxlen = self.list_len_largest_token[index]
                                                                          , padding = 'post'
                                                                          , value = 0.0).tolist())
                
                embedding_model = Sequential()
                embedding_model.add(Embedding(input_dim = vocab_size
                                , output_dim = 256
                                , name = "embedding"
                                , mask_zero = True
                                ))
                embedding_model.add(LSTM(LEN_VECTOR, name = "lstm"))
                embedding_model.compile(optimizer ='adam', loss = keras.losses.BinaryCrossentropy())
                                
                self.list_model[index] = embedding_model
                
                phaseC = time.time()
                print(f"embedding train complete: {col} {phaseC - start}")
                
        phase1 = time.time()
        print(f"embedding train complete: {phase1 - start}")
            
        self.save_model(distant_file=True)
        
        
    def run_model(self,df_path):
        """
        transforme les données de :param df_path: via les modèles entrainés

        :param df_path: fichier à traiter
        """
        
        start = time.time()
        
        flatten = lambda l: [item for sublist in l for item in sublist]
        
        new_name = incrementTmpFile(df_path)
        total_list = []
        for index, col in enumerate(self.list_df_cols):
            
            self.encoded_corpus[index] = np.array(self.encoded_corpus[index])
            
            print(f'{col} is being vectorized ...')
            print(f'self.encoded_corpus[index] is equal {len(self.encoded_corpus[index])}')
            full_col_vector = self.list_model[index].predict(self.encoded_corpus[index])                        
            print(f'shape is: {full_col_vector.shape}')
            self.encoded_corpus[index] = None
            if len(total_list) == 0:
                total_list = full_col_vector
            else:
                total_list = np.concatenate((total_list, full_col_vector), axis = 1)        
        save(f'tmp/{new_name}.npy', total_list)
                
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
                self.list_model[index].save(f"{self.id_run}_embedd_{self.list_df_cols[index]}.h5")
        else:
            for index,mod in enumerate(self.list_model):
                name_file = f"{self.id_run}_embedd_{self.list_df_cols[index]}.h5"
                self.list_model[index].save(name_file)
                save_file_on_minio(name_file, dir_name=self.path_run)
                os.remove(name_file)
                       
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