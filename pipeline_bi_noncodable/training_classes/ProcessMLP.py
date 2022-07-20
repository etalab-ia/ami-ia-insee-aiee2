"""
Classe qui se charge de process les données afin de les donner à un modèle keras.

Author : bsanchez@starclay.fr
date : 03/09/2020
"""
import tensorflow as tf

import gensim
import tempfile
import pickle
from datetime import datetime
from . import utils
import numpy as np
import pandas as pd
import logging

import tensorflow as tf
from tensorflow.keras.models import load_model

from .Model import *

from gensim.corpora import Dictionary
from nltk.probability import FreqDist
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.utils import class_weight

import tensorflow.keras.utils as np_utils
from sklearn.preprocessing import LabelEncoder

import glob
import os
import shutil
import pickle
import fasttext
import fasttext.util

from .Process import Process

import scipy.sparse


class ProcessMLP(Process):
    
    def __init__(self,  path_run, id_run, list_cols, dict_models={}, fasttext=False, embeddings_size=296, list_df_cols=None):
        """
        Classe permettant de transformer les inputs textuels en vecteurs
        pour un modèle Keras

        :param list_df_cols: liste des colonnes à traiter
        :param path_run: chemin de svg sur minio
        :param id_run: id du run de training
        :param dict_models: dictionnaires modèles (list of dict)
        :param fasttext: bool. Si true, utilise les embeddings préentrainés fasttext
        :param list_cols: liste des colonnes à traiter
        """
        self.fasttext = fasttext
        self.list_cols = list_cols
        self.tokenizer = None
        self.dict_info = None
        self.embeddings_size = embeddings_size

    def train_model(self, df_path):
        """
        L'objectif est partir d'un série de donnée de type [["groupe pomona cltinconnue 32 boulevard Hausseman"], ["rsxinconnue cltinconnue  infirmière ..."]]
        à une séquence tokeniser, encodé et paddé de ce type [[1, 2, 3 ,0 ,0 ,0 ,48, 658, 0, 0 ,0], [7, 9, 3 ,66 ,0 ,0 ,48, 74, 12, 0 ,0] ...].
        
        Le padding est réalisé par colonnes et concatené.
        
        Le vocabulaire est partagé sur toutes les colonnes.
        
        Les information récolté sur le dataset (taille du vocabulaire, taille max des token ...) est transmis au modèle keras
        
        :param df_path: fichier à traiter
        """
        tf.compat.v1.disable_eager_execution() # MEMORY LEAK OTHERWISE 

        mydir = os.path.dirname(os.path.realpath(__file__))
        
        flatten = lambda l: [item for sublist in l for item in sublist]

        #########################
        # Preparation donnée
        #########################
        
        X = pd.read_csv(df_path, sep=';', dtype=str)
        
        list_df_cols = ["fusion"]

        dico_vocab = []
        list_model = []
        len_largest_token = -1
        encoded_corpus = []
        vocab_size = -1
        list_embedd_matrix = []
                
        for col in list_df_cols:
            list_model.append(None)
            dico_vocab.append(Dictionary())
            encoded_corpus.append([])
            list_embedd_matrix.append(None)
        
        list_len_tokens_col = []
        
        for col in self.list_cols:
            list_len_tokens_col.append(-1)
            
        X = X[self.list_cols]
        
         # On "fusionne" toutes les sources de données dans un même champs
        X['fusion'] = X['rs_x'] + " " + X['clt_x'] + " " + X['profs_x'] + "  " + X['profi_x'] + " " + X['profa_x'] + " " + X['numvoi_x'] + " " + X['typevoi_x'] + "  " + X['actet_x'] + " " + X['dlt_x'] + " " + X['plt_x'] + " " + X['vardompart_x'] 
            
        train_corpus = X['fusion'].values.tolist()
        nb_docs = len(train_corpus)

        fdist = FreqDist()

        #Determiner la plus grande list de token 
        for index_doc, doc in enumerate(train_corpus):
            tokens = str(doc).split(" ")
            dico_vocab[0].add_documents([tokens])
            for token in tokens:
                fdist[str(token).lower()] += 1
            if(len_largest_token < len(tokens)):
                len_largest_token = len(tokens)

        more_than_one = list(filter(lambda x: x[1] >= 2, fdist.items()))
        vocab_size = len(dico_vocab[0])

        for index_doc, doc in enumerate(train_corpus):
                tokens = str(doc).split(" ")
                train_corpus[index_doc] = tokens
        self.tokenizer = Tokenizer(num_words = vocab_size, lower=True, char_level=False)
        self.tokenizer.fit_on_texts(train_corpus)
                        
        list_padding = []
            
        for col in self.list_cols:
            list_padding.append(0)
            
        for i, col in enumerate(self.list_cols):
            X[col] = X[col].astype(str)
            train_col_corpus = X[col].values.tolist()
            for ir, d in enumerate(train_col_corpus):
                tokens = str(d).split(" ")
                train_col_corpus[ir] = tokens
                for token in tokens:
                    if(list_len_tokens_col[i] < len(tokens)):
                        list_len_tokens_col[i] = len(tokens)
            word_seq_train = self.tokenizer.texts_to_sequences(train_col_corpus)
            for x in range(len(word_seq_train)):
                word_seq_train[x] = flatten(pad_sequences([word_seq_train[x]]
                                                                          , maxlen = list_len_tokens_col[i]
                                                                          , padding = 'post'
                                                                          , value = 0.0).tolist())
            list_padding[i] = word_seq_train
                
        # On concatenne les colonnes padder
        word_seq_train  = np.concatenate((list_padding), axis = 1)
        encoded_corpus[0] = word_seq_train
           
            # Montre le nb de mots non trouvé, au minimun 1 à cause du padding '0'
#         df_encoded = pd.DataFrame({"fusion":pd.Series(list(encoded_corpus[0])) })
        
        new_name = incrementTmpFile(F"{df_path}")
#         df_encoded = df_encoded.values
        np.save(f"tmp/{new_name}.npy",encoded_corpus[0])  

        self.dict_info = {
            "n_features" : self.embeddings_size,
            "len_largest_token" : len_largest_token,
            "vocab_size" : vocab_size,
            "list_len_tokens_col" : list_len_tokens_col,
            "fasttext": self.fasttext
        }
        parent_path = os.path.dirname(os.path.normpath(mydir))
        os.makedirs(os.path.join(parent_path, "output"), exist_ok=True)
        with open(os.path.join(parent_path, "output", "dict_info.p"), 'wb') as file:
            pickle.dump(self.dict_info, file, protocol = pickle.HIGHEST_PROTOCOL)
            
        with open(os.path.join(parent_path, "output", "tokenizer.p"), 'wb') as file:
            pickle.dump(self.tokenizer, file, protocol = pickle.HIGHEST_PROTOCOL)          
        
        
        labels = scipy.sparse.load_npz('sp_target.npz')
        labels = np.asarray(flatten(labels.toarray()))
        encoder = LabelEncoder()
        encoder.fit(labels)
        encoded_Y = encoder.transform(labels)
        labels = np_utils.to_categorical(encoded_Y)
        np.save('target.npy', labels)
        
        if self.fasttext:
            # On remplis la matrice d'embedding fastext
            # Les mots non trouvé auront un vecteur null
            logger = logging.getLogger('fasttext')
            fasttext_size = self.embeddings_size # Must be divisible by number of attention head
            fasttext_path = os.path.join(os.path.dirname(mydir), 'fasttext', f'cc.fr.{fasttext_size}.bin')
            ft_model=None
            if not os.path.exists(fasttext_path):
                logger.info('Downloading fasttext from source...')
                os.makedirs(os.path.dirname(fasttext_path), exist_ok=True)
                fasttext.util.download_model('fr', 'strict')
                shutil.move('cc.fr.300.bin', os.path.join(os.path.dirname(fasttext_path), 'cc.fr.300.bin'))
                if fasttext_size != 300:
                    logger.info(f'Resizing fasttext to {fasttext_size}...')
                    ft_model = fasttext.load_model(os.path.join(os.path.dirname(fasttext_path), 'cc.fr.300.bin'))
                    fasttext.util.reduce_model(ft_model, fasttext_size)
                    ft_model.save_model(fasttext_path)
            if ft_model is None:
                logger.info(f'loading fasttext, size {fasttext_size}...')
                ft_model = fasttext.load_model(fasttext_path)
            
            words_not_found = []
            embedding_matrix = np.zeros((vocab_size + 1, self.embeddings_size))
            for word, i in self.tokenizer.word_index.items():
                embedding_vector = ft_model[word]
                if (embedding_vector is not None) and len(embedding_vector) > 0:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector
                else:
                    words_not_found.append(word)
            list_embedd_matrix[0] = embedding_matrix 
            np.save('embedding_fasttext.npy', embedding_matrix)
        
    def run_model(self, df_path, path_tokeniser=None, path_dictinfo=None, save_dir='tmp'):
        """
        Encode les données en fonction des fichiers "output" (voir readme) sirué dans input/
        
        :param df_path: fichier à traiter
        :param path_tokeniser: chemin vers la svg du tokenizer s'il n'est pas chargé
        :param path_dictinfo: chemin vers la svg du dict_info s'il n'est pas chargé
        :param save_dir: dossier ou sauvegarder l'output
        """
        chunksize = 60000
        
        flatten = lambda l: [item for sublist in l for item in sublist]
        
        # Load pre-trained encoder
        if self.tokenizer is None:
            if path_tokeniser is None:
                parent_path = os.path.dirname(os.path.normpath(mydir))
                path_tokeniser = os.path.join(parent_path, "output", "tokenizer.p")
            with open(path_tokeniser, "rb") as input_file:
                self.tokenizer = pickle.load(input_file)
           
        if self.dict_info is None:
            if path_dictinfo is None:
                parent_path = os.path.dirname(os.path.normpath(mydir))
                path_dictinfo = os.path.join(parent_path, "output", "dict_info.p")
            with open(path_dictinfo, "rb") as input_file:
                self.dict_info = pickle.load(input_file)
        
        list_len_tokens_col = self.dict_info['list_len_tokens_col']


        transformed_data = []
        for X in pd.read_csv(df_path, sep=';', dtype=str, chunksize=chunksize):
            list_padding = []
                
            for col in self.list_cols:
                list_padding.append(0)
                
            for i, col in enumerate(self.list_cols):
                X[col] = X[col].astype(str)
                train_col_corpus = X[col].values.tolist()
                for ir, d in enumerate(train_col_corpus):
                    tokens = str(d).split(" ")
                    train_col_corpus[ir] = tokens
                word_seq_train = self.tokenizer.texts_to_sequences(train_col_corpus)
                for x in range(len(word_seq_train)):
                    if len(word_seq_train[x]) > list_len_tokens_col[i]:
                        word_seq_train[x] = word_seq_train[x][:list_len_tokens_col[i]]
                    word_seq_train[x] = flatten(pad_sequences([word_seq_train[x]], 
                                                              maxlen = list_len_tokens_col[i],
                                                              padding = 'post',
                                                              value = 0.0).tolist())
                list_padding[i] = word_seq_train
                    
            # On concatenne les colonnes padder
            transformed_data.append(np.concatenate((list_padding), axis = 1))

        new_name = incrementTmpFile(os.path.join(save_dir, os.path.basename(df_path)))
        np.save(os.path.join(save_dir, new_name + ".npy"), np.concatenate(transformed_data, axis=0)) 
            
        
    def apply(self,df_path):
        """
        extrait puis applique les modèles Doc2Vec sur les données

        :param df_path: fichier à traiter
        """
        self.train_model(df_path)
        # self.run_model(df_path)
        
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