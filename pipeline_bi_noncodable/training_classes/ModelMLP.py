"""
Classe de modèle

Author : bsanchez@starclay.fr
date : 06/08/2020
"""
import tensorflow as tf

import gensim
import tempfile
import pickle
from datetime import datetime
from . import utils
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model

from gensim.corpora import Dictionary
from nltk.probability import FreqDist
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.utils import class_weight

import tensorflow.keras.utils as np_utils
from sklearn.preprocessing import LabelEncoder

from .Model import *

class MLP(Model):
    
    def __init__(self, id_run, path_run):
        """
        Modèle Deep Learning à base de LSTM bidirectionnelles
        Chaque champ est un input du modèle
        """
        super().__init__(name="mlp"
                         ,model = None
                         ,id_run = id_run
                         ,path_run = path_run)
        
    def train_model(self, X, labels):
        tf.compat.v1.disable_eager_execution() # MEMORY LEAK OTHERWISE 
        flatten = lambda l: [item for sublist in l for item in sublist]

        
        #########################
        # Preparation donnée
        #########################
        
        list_df_cols = [ "rs_x"
                        ,"clt_x"
                        ,"profs_x"
                        ,"profi_x"
                        ,"profa_x"
                        ,"numvoi_x"
                        ,"typevoi_x"
                        ,"actet_x"
                        ,"dlt_x"
                        ,"plt_x"
                        ,"vardompart_x" ]
        
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
        
        dico_vocab = []
        list_model = []
        list_len_largest_token = []
        encoded_corpus = []
        list_vocab_size = []
        print("begin pre-process...")
        for col in list_df_cols:
            list_vocab_size.append(0)
            list_model.append(None)
            dico_vocab.append(Dictionary())
            encoded_corpus.append([])
            list_len_largest_token.append(-1)
                
        for index, col in enumerate(list_df_cols):
            print(f'{col}')                   
            train_corpus = X[col].values.tolist()
            print(train_corpus[0:5])
            nb_docs = len(train_corpus)

            list_len_largest_token[index] = -1

            fdist = FreqDist()

            #Determiner la plus grande list de token 
            for index_doc, doc in enumerate(train_corpus):
                tokens = str(doc).split(" ")
                dico_vocab[index].add_documents([tokens])
                for token in tokens:
                    fdist[token.lower()] += 1
                if(list_len_largest_token[index] < len(tokens)):
                    list_len_largest_token[index] = len(tokens)

            more_than_one = list(filter(lambda x: x[1] >= 2, fdist.items()))

            list_vocab_size[index] = len(dico_vocab[index])

            print(f"vocab_size {list_vocab_size[index]} (filtré) vs {len(dico_vocab[index])}")

            print(f"largest token is: {list_len_largest_token[index]}")

            for x in train_corpus:
                encoded_corpus[index].append(one_hot(str(x),list_vocab_size[index]))
                
            del train_corpus
            
            print(encoded_corpus[index][0:5])
            
            for x in range(len(encoded_corpus[index])):
                encoded_corpus[index][x] = flatten(pad_sequences([encoded_corpus[index][x]]
                                                                              , maxlen = list_len_largest_token[index]
                                                                              , padding = 'post'
                                                                              , value = 0.0).tolist())
            print(encoded_corpus[index][0:5])
            print(f"type is {type(encoded_corpus)} and {type(encoded_corpus[index])} and {type(encoded_corpus[index][0])}")
        print(f'\n\nmerging encoded corpus...')                           
        df_encoded = pd.DataFrame()
        for index_col, col in enumerate(list_df_cols):
            df_encoded[col] = encoded_corpus[index_col]
            
        del encoded_corpus
        
        
        print('\n\ntrain test split...')   
        
        labels = np.asarray(flatten(labels.toarray()))
        encoder = LabelEncoder()
        encoder.fit(labels)
        encoded_Y = encoder.transform(labels)
        labels = np_utils.to_categorical(encoded_Y)
        print(labels)
    
        X_train, X_test, y_train, y_test = train_test_split(df_encoded
                                                     , labels
                                                     , test_size = 0.5
                                                     , random_state = 42
                                                     , stratify = labels)
        
        X_train, X_eval, y_train, y_eval = train_test_split(X_train
                                                            , y_train
                                                            , test_size = 0.3
                                                            , random_state = 42
                                                            , stratify = y_train)
        
        print(f"X_train shape {X_train.shape}")
        print(f"X_test shape {X_test.shape}")
        print(f"X_eval shape {X_eval.shape}")
        ########
        # Model
        ########
        
#         print(flatten(y_train.tolist()))
        class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(flatten(y_train.tolist())),
                                                 flatten(y_train.tolist()))
        print(class_weights)
        
        dict_class_weight = dict(enumerate(class_weights))
        
        
        
        # plusieurs colonne -> plusieurs input
        list_input = []
        list_embedd = []
        print('begin model creation...')    
        for index_col, col in enumerate(list_df_cols):
            print(f'{col}')    
            LEN_VECTOR = dict_len_col[col]
            
            print('*****************\n\n')
            
            input_shape = layers.Input(shape = list_len_largest_token[index_col], name = f"{col}_input")
            list_input.append(input_shape)
        
            embedd_layer = layers.Embedding(input_dim = list_vocab_size[index_col]
                                    , output_dim = 128
                                    , name = f"{col}_embedding"
                                    , mask_zero = True)(input_shape)
            
#             embedd_layer = layers.LSTM(LEN_VECTOR,  name = f"{col}_lstm")(embedd_layer) 
            embedd_layer = layers.Bidirectional(layers.LSTM(128, return_sequences=True))(embedd_layer)
            embedd_layer = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(embedd_layer)
#             embedd_layer = layers.GlobalAveragePooling1D()(embedd_layer)
#             embedd_layer = layers.Bidirectional(layers.LSTM(64, return_sequences=False))(embedd_layer)
            list_embedd.append(embedd_layer)

        
        dl_model = layers.concatenate([x for x in list_embedd])

        dl_model = layers.Dense(64, activation = 'relu', kernel_regularizer = regularizers.l2(0.0001))(dl_model)
        dl_model = layers.Dropout(0.3)(dl_model)
        
        
        dl_model = layers.Dense(32, activation = 'relu', kernel_regularizer = regularizers.l2(0.0001))(dl_model)
        dl_model = layers.Dropout(0.3)(dl_model)
        
        output = (layers.Dense(2, activation='softmax'))(dl_model)
        
        my_model = keras.Model(inputs=[x for x in list_input], outputs = output)
        
        print(my_model.summary())
                
        my_model.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy())
        
        mc = ModelCheckpoint('tmp/best_model.h5', monitor='val_loss', mode='min', verbose=1)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50)
        

        my_model.fit({ 'rs_x_input':        np.asarray(X_train["rs_x"].values.tolist())
                     , 'clt_x_input':       np.asarray(X_train["clt_x"].values.tolist())
                     , 'profs_x_input':     np.asarray(X_train["profs_x"].values.tolist())
                     , 'profi_x_input':     np.asarray(X_train["profi_x"].values.tolist())
                     , 'profa_x_input':     np.asarray(X_train["profa_x"].values.tolist())
                     , 'numvoi_x_input':    np.asarray(X_train["numvoi_x"].values.tolist())
                     , 'typevoi_x_input':   np.asarray(X_train["typevoi_x"].values.tolist())
                     , 'actet_x_input':     np.asarray(X_train["actet_x"].values.tolist())
                     , 'dlt_x_input':       np.asarray(X_train["dlt_x"].values.tolist())
                     , 'plt_x_input':       np.asarray(X_train["plt_x"].values.tolist())
                     , 'vardompart_x_input':np.asarray(X_train["vardompart_x"].values.tolist())}, y_train
                ,epochs = 4000
                ,batch_size = 1024
                ,validation_data = ({ 'rs_x_input':      np.asarray(X_eval["rs_x"].values.tolist())
                     , 'clt_x_input':                    np.asarray(X_eval["clt_x"].values.tolist())
                     , 'profs_x_input':                  np.asarray(X_eval["profs_x"].values.tolist())
                     , 'profi_x_input':                  np.asarray(X_eval["profi_x"].values.tolist())
                     , 'profa_x_input':                  np.asarray(X_eval["profa_x"].values.tolist())
                     , 'numvoi_x_input':                 np.asarray(X_eval["numvoi_x"].values.tolist())
                     , 'typevoi_x_input':                np.asarray(X_eval["typevoi_x"].values.tolist())
                     , 'actet_x_input':                  np.asarray(X_eval["actet_x"].values.tolist())
                     , 'dlt_x_input':                    np.asarray(X_eval["dlt_x"].values.tolist())
                     , 'plt_x_input':                    np.asarray(X_eval["plt_x"].values.tolist())
                     , 'vardompart_x_input':             np.asarray(X_eval["vardompart_x"].values.tolist())}, y_eval)
                ,class_weight = dict_class_weight
                ,callbacks = [es, mc]
                ,verbose = True)
        
        self.model =  load_model('tmp/best_model.h5')
        
        predictions = self.model.predict_classes(X_test)
        
        super().compute_metrics(y_test, predictions)
        
    def run_model(self,  X_test,  y_test):
        print("début prédiction")
        
        predictions = self.model.predict_classes(X_test)        

        print("fin prédiction")
        super().compute_metrics(y_test, predictions)
        
        
    def super_apply(self, X, labels):
        """
        Method qui gère le workflow au sein du modèle (train -> test -> sauvegarde)
                
        :params X_train: matrice creuse contenant les donnée d'entrainement
        :params y_train: Matrice pleine contenant les labels
        :params X_test: matrice creuse contenant les donnée de test
        :params y_test: Matrice pleine contenant les label
    
        :returns: void
        """
        self.train_model(X, labels)
        print('fin apply')

    def objectiv_function_mlp(self,space,X_train,y_train):
        scores = []
        kf = KFold(n_splits=10)
        kf.get_n_splits(X_train)
        KFold(n_splits=2, random_state=None, shuffle=False)

        nb_neurone_first_layer = space['nb_neurone_first_layer']
        nb_neurone_second_layer = space['nb_neurone_second_layer']

        for train_index, test_index in kf.split(X_train):
            Xcv_train, Xcv_test = X_train[train_index],X_train[test_index]
            ycv_train, ycv_test = y_train[train_index],y_train[test_index]

            mlp = keras.Sequential()
            mlp.add(layers.Dense(Xcv_train.shape[1],activation='relu',input_dim=Xcv_train.shape[1]))
            mlp.add(layers.Dropout(0.3))

            mlp.add(layers.Dense(int(nb_neurone_first_layer),activation='relu'))
            mlp.add(layers.Dropout(0.3))

            mlp.add(layers.Dense(1,activation='relu'))

            mlp.compile(optimizer='adam',loss=keras.losses.BinaryCrossentropy())
            mlp.fit(Xcv_train,ycv_train,epochs=1000,batch_size=512,verbose=False)

            scores.append(brier_score_loss(ycv_test,mlp.predict_proba(Xcv_test)[:,1]))

        loss = np.mean(scores)

        return np.mean(loss)