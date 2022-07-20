"""
Classe qui se charge de créer des embeddings fasttext

Author : bsanchez@starclay.fr
date : 06/08/2020
"""
from tqdm import tqdm

import tempfile
import pickle
from datetime import datetime

import pandas as pd
import numpy as np
from numpy import asarray
from numpy import save
from numpy import load

import os
import s3fs
from .Process import Process
from .utils import *
import scipy
from string import punctuation
import time

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from nltk import WordPunctTokenizer
from nltk.tokenize import word_tokenize

import gensim
from gensim.corpora import Dictionary
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models.phrases import Phrases, Phraser
from gensim.models.fasttext import FastText

import fasttext
from fasttext import load_model


#keras
import tensorflow.keras as keras
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Embedding, Input, LSTM
from tensorflow.keras.models import Model

from nltk.probability import FreqDist

# https://github.com/keras-team/keras/issues/13118
import tensorflow.keras.backend as K


class ProcessFastText(Process):
    
    def __init__(self, list_df_cols, path_run, id_run, dict_models={}):
        """
        Classe permettant de transformer les inputs textuels en vecteurs
        via un modèle d'embeddings fasttext preentrainé

        :param list_df_cols: liste des colonnes à traiter
        :param path_run: chemin de svg sur minio
        :param id_run: id du run de training
        :param dict_models: dictionnaires modèles (list of dict)
        """
        self.name = "EmbeddingFastText"
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
        
        Les embedding sont pris depuis un fichier fasttext ainsi le layer embedding n'est pas entrainable.
        
        Retourne un fichier npy contenant tout les embedding

        :param df_path: fichier à traiter
        """
        start = time.time()
        
        flatten = lambda l: [item for sublist in l for item in sublist]
        
        print("Loading FastText model ...")
        ft_model = load_model('cc.fr.300.bin')
        print("Loaded FastText model ...")
        n_features = ft_model.get_dimension()
            
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
                vocab_size = len(self.dico_vocab[index]) + 1 # +1 car on compte le padding => "la valeur '0' "
                # On transforme la liste de mot en list de list de token
                list_list_tokens = []
                for index_doc, doc in enumerate(train_corpus):
                    tokens = doc.split(" ")
                    train_corpus[index_doc] = tokens
                tokenizer = Tokenizer(num_words = vocab_size, lower=True, char_level=False)
                tokenizer.fit_on_texts(train_corpus)
                
                # Les token sont remplacé par leur "id" chien devient '2'*
                # On utilise le tokeniser nltk car les id commence à 1 et non à 0 comme celuit de gensim
                # autrement on aura un problème de padding / embedding
                word_seq_train = tokenizer.texts_to_sequences(train_corpus)
                print(self.list_len_largest_token[index])
                # Les list de token sont paddé
                for x in range(len(word_seq_train)):
                    word_seq_train[x] = flatten(pad_sequences([word_seq_train[x]]
                                                                          , maxlen = 5
                                                                          , padding = 'post'
                                                                          , value = 0.0).tolist())
               
                self.encoded_corpus[index] = word_seq_train
                # On remplis la matrice d'embedding fastext
                # Les mots non trouvé auront un vecteur null
                words_not_found = []
                embed_dim = 300
                embedding_matrix = np.zeros((vocab_size, embed_dim))
                for word, i in tokenizer.word_index.items():
                    embedding_vector = ft_model[word]
                    if (embedding_vector is not None) and len(embedding_vector) > 0:
                        # words not found in embedding index will be all-zeros.
                        embedding_matrix[i] = embedding_vector
                    else:
                        words_not_found.append(word)
                        
                # Montre le nb de mots non trouvé, au minimun 1 à cause du padding '0'
                print(f'number of null word embeddings: {np.sum(np.sum(embedding_matrix, axis=1) == 0)}')
                embedding_model = Sequential()
                embedding_model.add(Embedding(input_dim = vocab_size
                                , output_dim = 300
                                , name = "embedding"
                                , mask_zero = True
                                , weights = [embedding_matrix]
                                , trainable = False
                                ))
                embedding_model.add(LSTM(LEN_VECTOR, name = "lstm"))
                embedding_model.compile(optimizer ='adam', loss=keras.losses.BinaryCrossentropy())
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

        :param df_path: fichier à traiter
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