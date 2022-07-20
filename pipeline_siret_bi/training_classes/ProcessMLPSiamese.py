"""
Modèle de similarité :
On entraine un embedding personalisé qui permet de plonger les BI et les Sirus dans un même espaces de représentation.
Afin de matcher un bi, on regarde les sirus les plus simulaires (cosine similarity).

Afin d'entrainer les modèles, il est nécéssaire d'encoder les données, c'est ce que fait ce fichier

Author : bsanchez@starclay.fr
date : 23/09/2020
"""
import os
import sys
import pickle
import shutil
import logging

import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.text import Tokenizer

from gensim.corpora import Dictionary
import fasttext
import fasttext.util
from nltk import ngrams

from .process import Process


class ProcessMLPSiamese(Process):

    def __init__(self, local_path_run, list_cols, 
                 fasttext=False, 
                 fasttext_size=296,
                 ngram: str = "3"):
        """
        Organise le preprocess du modèle de similarité.
        
        Input: DataFrame nettoyé
        
        On calcule le vocabulaire et le tokeniser en passant sur tout les mots du jeu de donnée.
        Puis on encode via le tokeniser chaque partie du dataset : la partie BI et la partie Sirus
        
        Les résultats sont stocké dans une matrice à 2 entrée : une matrice bi et une matrice sirus
        
        Pour des soucis de taille de mémoire, les fichiers output sont dumpé via pickle.
        
        Output : siamese_corpus Matrice de donnée encodée
        dict_info: un dictionnaire avec des information sur le dataset. Le dictionnaire sera utilisé par le modèle de similarité.
        tokeniser: tokeniser keras dumper dans un fichier pickle
        
        Ex: print(siamese_corpus[0])
        >> [[0],[68,741,38], [99,75]
        >> ,[0,0],[741,38], []
        >> ...]
        
        Chaque ligne de la matrice est une ligne du dataframe. Toutes les lignes ont un 
        nombre d'élément fixe (colonne du dataframe d'entrée). Les données dans les colonnes ne sont pas pas paddé et sont donc de taille variable.
        Deux matrices sont produite, BI et sirus, et sont stocké dans une liste (a deux entrée donc).
        
        :param local_path_run: chemin local du training en cours
        :param list_cols: dict de listes
                            - type_de_données: [colonnes à traiter]
        :param fasttext: bool (utilisation de fasttext ou non) (uniquement si ngram='word')
        :param fasttext_size: taille des embeddings
        :param ngram: str. Si entier ('3'), taille des ngrams. Si 'word', on tokenise par mot
        """
        self.dir_path = local_path_run
        self.tokenizer_path = os.path.join(self.dir_path, "tokenizer.p")
        self.dict_info_path = os.path.join(self.dir_path, 'dict_info.pickle')
        self.fasttext_emb_path = os.path.join(self.dir_path, 'embedding_fasttext.npy')
        self.fasttext = fasttext
        self.fasttext_size = fasttext_size
        self.list_cols = list_cols
        self.ngram = ngram
        if self.fasttext and self.ngram != 'word':
            raise ValueError('Fasttext is True, ngram should be "word"')

    def train(self, input_df_path):        
        """
        Entraine le tokenizer, et (si besoin) extrait les embeddings fasttext

        :param input_df_path: chemin du fichier à traiter
        """
        ####
        # INIT
        ###
        len_full_largest_token : int =  -1 # plus grande liste de  token présent dans le jeu de donnée (sirus + bi) 
        vocab_size_full : int = -1 # nombre de token différent
        chunksize : int  = 60000 # longeur du block chargé, à augmenter si beaucoup de mémoire
            
        if self.ngram.isdigit():            
            split_method = lambda x: list(ngrams(x, int(self.ngram)))
        elif self.ngram == "word":
            split_method = lambda x: x.split()
        else:
#             print("Erreur sur l'argument ngram")
            sys.exit(2)

        #########################
        # Preparation donnée
        # On update les métriques (longeur token vocabulaire ...)
        #########################
        tokenizer = Tokenizer(lower=True, char_level=False, oov_token='oov_token')
        # TODO tokenizer looses tokens...
        for X in pd.read_csv(input_df_path, sep=';', dtype=str, chunksize = chunksize):
            for col in X:
                X[col] = X[col].astype(str)  
             
            # On "fusionne" toutes les sources de données dans un même champs pour aller plus vite
            X['fusion_bi'] = X[self.list_cols['fusion_bi']].agg(' '.join, axis=1) 
            X['fusion_si'] = X[self.list_cols['fusion_si']].agg(' '.join, axis=1)
            train_corpus = []
            for _, doc in enumerate(X['fusion_bi'].values.tolist() + X['fusion_si'].values.tolist()):
                tokens = split_method(doc)
                tokens = [''.join(t) for t in tokens]
                if len_full_largest_token < len(tokens):
                    len_full_largest_token = len(tokens)
                train_corpus.append(tokens)
            tokenizer.fit_on_texts(train_corpus)

        vocab_size_full = len(tokenizer.word_index) + 2 # +1 pour marquer le 0 et +1 par sécurité
        tokenizer.num_words = len(tokenizer.word_index)  # pour garder tous les mots
        
        with open(self.tokenizer_path, "wb") as f:
            pickle.dump(tokenizer, f)
        # Nécéssaire pour le modèle de similarité
        dict_info = {
                "n_features" : self.fasttext_size,
                "len_full_largest_token" : len_full_largest_token,
                "vocab_size_full" : vocab_size_full,
                "ngram" : self.ngram
            }
        with open(self.dict_info_path, 'wb') as file:
            pickle.dump(dict_info, file, protocol = pickle.HIGHEST_PROTOCOL)

        if self.fasttext:
            # load fasttext
            logger = logging.getLogger('fasttext')
            mydir = os.path.dirname(os.path.realpath(__file__))
            fasttext_path = os.path.join(os.path.dirname(mydir), 'fasttext', f'cc.fr.{self.fasttext_size}.bin')
            ft_model=None
            if not os.path.exists(fasttext_path):
                logger.info('Downloading fasttext from source...')
                os.makedirs(os.path.dirname(fasttext_path), exist_ok=True)
                fasttext.util.download_model('fr', 'strict')
                shutil.move('cc.fr.300.bin', os.path.join(os.path.dirname(fasttext_path), 'cc.fr.300.bin'))
                if self.fasttext_size != 300:
                    logger.info(f'Resizing fasttext to {self.fasttext_size}...')
                    ft_model = fasttext.load_model(os.path.join(os.path.dirname(fasttext_path), 'cc.fr.300.bin'))
                    fasttext.util.reduce_model(ft_model, self.fasttext_size)
                    ft_model.save_model(fasttext_path)
            if ft_model is None:
                logger.info(f'loading fasttext, size {self.fasttext_size}...')
                ft_model = fasttext.load_model(fasttext_path)
            # get embeddings
            words_not_found = []
            embed_dim = self.fasttext_size
            embedding_matrix = np.zeros((vocab_size_full + 1, embed_dim))
            for word, i in tokenizer.word_index.items():
                embedding_vector = ft_model[word]
                if (embedding_vector is not None) and len(embedding_vector) > 0:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector
                else:
                    words_not_found.append(word)
            # Montre le nb de mots non trouvé, au minimun 1 à cause du padding '0'
#             print(f'number of null word embeddings: {np.sum(np.sum(embedding_matrix, axis=1) == 0)}')
            np.save(self.fasttext_emb_path, embedding_matrix)
            
###############################################################################################################################
                
    def run(self, input_df_path, output_file):
        """
        Encode les données (doit être entrainé avant)

        :param input_df_path: chemin du fichier d'entrée
        :param output_file: chemin du fichier de sortie
        
        """
        chunksize = 60000
        
        # Load pre-trained encoder
        with open(self.tokenizer_path, "rb") as input_file:
            tokenizer = pickle.load(input_file)
           
        # with open(path_tokeniser, "rb") as input_file:
        #     dict_info = pickle.load(input_file)
            
        if self.ngram.isdigit():            
            split_method = lambda x: list(ngrams(x, int(self.ngram)))
        elif self.ngram == "word":
            split_method = lambda x: x.split()
        else:
            print("Erreur sur l'argument ngram")
            sys.exit(2)
        
        siamese_corpus = []
        for X in pd.read_csv(input_df_path, sep=';', dtype=str, chunksize = chunksize):
#             print(f"Chunksize is {X.shape}")
            for col in X:
                X[col] = X[col].astype(str)
                
            df_corpus = []
            for index, siam in enumerate(self.list_cols):
#                 print(f"\n** {siam} **")
                list_encoded_cols = []
                for col in self.list_cols[siam]:
#                     print(f"__{col}__")
                    corpus = X[col].values.tolist()
#                     print(corpus[0:2])
                    for i, row in enumerate(corpus):
                        tokens = split_method(row)
                        while not len(tokens) and len(row):
                            # input is too short to make a token => we pad it with ' '
                            row = ' ' + row
                            tokens = split_method(row)
                        tokens = [''.join(t) for t in tokens]
                        corpus[i] = tokens
#                     print(corpus[0:2])
                    word_seq_train = tokenizer.texts_to_sequences(corpus)
#                     print(word_seq_train[0:2])
                    list_encoded_cols.append(word_seq_train)
                encoded_anchor = []
                for i in range(len(list_encoded_cols[0])):
                    encoded_anchor.append([col[i] for col in list_encoded_cols])
#                 print("______")
#                 print(encoded_anchor[0:2])
#                 print("______")
                df_corpus.append(encoded_anchor)
            siamese_corpus.append(df_corpus)

        # reformat into [X, Y]
        siamese_corpus = [sum([d[i] for d in siamese_corpus], []) for i in range(len(self.list_cols))]
        if output_file[-4:] == '.npy':
            np.save(output_file, siamese_corpus)
        elif output_file[-2:] == '.p':
            pickle.dump(siamese_corpus, open(output_file, "wb" ))
        else:
            raise ValueError(f'Format not recognized: {output_file}')
        
