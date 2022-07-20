"""
Classe de modèle Transfomer 

https://keras.io/examples/nlp/text_classification_with_transformer/

Author : bsanchez@starclay.fr
date : 04/09/2020
"""
import gensim
import tempfile
import pickle
from datetime import datetime
from . import utils
import numpy as np
import logging

from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import balanced_accuracy_score

from sklearn.metrics import confusion_matrix
from sklearn.calibration import calibration_curve

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model

from sklearn.model_selection import train_test_split

from .Model import *

from gensim.corpora import Dictionary
from nltk.probability import FreqDist
from tensorflow.keras.preprocessing.text import one_hot, Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

from sklearn.utils import class_weight

import tensorflow.keras.utils as np_utils
from sklearn.preprocessing import LabelEncoder

import glob
import os

import pickle

import fasttext
import fasttext.util


class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8,**kwargs):
        super(MultiHeadSelfAttention, self).__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
        })
        return config

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(query, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(key, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(value, batch_size)  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(concat_attention)  # (batch_size, seq_len, embed_dim)
        return output
    

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        self.att, self.ffn = None, None
        self.layernorm1, self.layernorm2 = None, None
        self.dropout1, self.dropout2 = None, None

        for att_name in ['att', 'ffn', 'layernorm1', 'layernorm2', 'dropout1', 'dropout2']:
            if att_name in kwargs:
                setattr(self, att_name, kwargs[att_name])
                del kwargs[att_name]

        super(TransformerBlock, self).__init__(**kwargs)
        if self.att is None:
            self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        if self.ffn is None:
            self.ffn = keras.Sequential(
                [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
            )
        if self.layernorm1 is None:
            self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        if self.layernorm2 is None:
            self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        if self.dropout1 is None:
            self.dropout1 = layers.Dropout(rate)
        if self.dropout2 is None:
            self.dropout2 = layers.Dropout(rate)
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'layernorm1': self.layernorm1,
            'layernorm2': self.layernorm2,
            'dropout1': self.dropout1,
            'dropout2': self.dropout2
        })
        return config

    def call(self, inputs):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=True)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=True)
        return self.layernorm2(out1 + ffn_output)


class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim,  list_len_tokens_col, pre_trained_weights = None, **kwargs):
        self.token_emb = None
        self.pos_emb = None
        self.col_pos_emb = None
        if 'token_emb' in kwargs:
            self.token_emb = kwargs['token_emb']
            del kwargs['token_emb']
        if 'pos_emb' in kwargs:
            self.pos_emb = kwargs['pos_emb']
            del kwargs['pos_emb']
        if 'col_pos_emb' in kwargs:
            self.col_pos_emb = kwargs['col_pos_emb']
            del kwargs['col_pos_emb']
        super(TokenAndPositionEmbedding, self).__init__(**kwargs)
        self.list_len_tokens_col = list_len_tokens_col
        
        if self.token_emb is None:
            if pre_trained_weights is None:
                self.token_emb = layers.Embedding(input_dim = vocab_size + 1, 
                                                  output_dim = embed_dim,
                                                  mask_zero = True,
                                                  name='token_emb')
            else:
                self.token_emb = layers.Embedding(input_dim = vocab_size + 1,
                                                  output_dim = embed_dim,
                                                  weights = pre_trained_weights,
                                                  trainable = True,
                                                  mask_zero = True,
                                                  name='token_emb')
        if self.pos_emb is None:
            self.pos_emb = layers.Embedding(input_dim = np.sum(self.list_len_tokens_col), output_dim = embed_dim, name='pos_emb')
        if self.col_pos_emb is None:
            self.col_pos_emb = layers.Embedding(input_dim = 11, output_dim = embed_dim, name='col_pos_emb') # 11 = nb columns
        
    def get_config(self):
        config = super(TokenAndPositionEmbedding, self).get_config().copy()
        config.update({
            'token_emb': self.token_emb,
            'pos_emb': self.pos_emb,
            'col_pos_emb': self.col_pos_emb
        })
        return config

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start = 0, limit = np.sum(self.list_len_tokens_col), delta = 1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        fields_representation = np.array([])
        for index_fields, len_tokens in enumerate(self.list_len_tokens_col):
            fields_representation = np.concatenate((fields_representation, np.full(shape = len_tokens, fill_value = index_fields, dtype = np.int)), axis = None) 
        col_pos = self.col_pos_emb(fields_representation)
        return  positions + col_pos + x 
    

class TransformerModel(tf.keras.Model):

    def __init__(self, n_features, len_largest_token, vocab_size, list_len_tokens_col, 
                 nb_transformerblocks=1, num_heads=8, transformer_ffdim=64, denses_sizes=[256, 64, 32],
                 list_embedd_matrix=None):
        super(TransformerModel, self).__init__()

        embed_dim = n_features  # Embedding size for each token
        # num_heads = 8  # Number of attention heads
        # ff_dim = 64  # Hidden layer size in feed forward network inside transformer

        # inputs = layers.InputLayer(shape=(np.sum(list_len_tokens_col).item()))
        logging.info(f'Transformer conf: {nb_transformerblocks}/{num_heads}/{transformer_ffdim}/{denses_sizes}')
        if list_embedd_matrix is not None:
            self.embedding_layer = TokenAndPositionEmbedding(len_largest_token, vocab_size, embed_dim, list_len_tokens_col, [list_embedd_matrix],
                                    input_shape=(np.sum(list_len_tokens_col).item(),))
        else:
            self.embedding_layer = TokenAndPositionEmbedding(len_largest_token, vocab_size, embed_dim, list_len_tokens_col,
                                    input_shape=(np.sum(list_len_tokens_col).item(),))

        self.transformers_blocks = [TransformerBlock(embed_dim, num_heads, transformer_ffdim) for _ in range(nb_transformerblocks)]
        self.transformer_normalizations = [layers.LayerNormalization(epsilon=1e-6) for _ in range(nb_transformerblocks)]
        self.global_pooling = layers.GlobalAveragePooling1D()
        self.dropout_gp = layers.Dropout(0.1)
        self.denses = [layers.Dense(s, activation="relu") for s in denses_sizes]
        self.denses_dropout = [layers.Dropout(0.1) for _ in denses_sizes]
        self.output_dense = layers.Dense(2, activation='softmax')

    def call(self, inputs):
        x = self.embedding_layer(inputs) 
        for block, norm in zip(self.transformers_blocks, self.transformer_normalizations):
            x = norm(x+block(x))
            
        x = self.global_pooling(x)
        x = self.dropout_gp(x)
        
        for l, d in zip(self.denses, self.denses_dropout):
            x = d(l(x))

        return self.output_dense(x)


class MLPTransformer(Model):
    
    def __init__(self, id_run, path_run, fasttext = False, nb_transformerblocks=1, num_heads=8, transformer_ffdim=64, denses_sizes=[256, 64, 32]):
        """
        Modèle basé sur le modèle Deep Learning Transformer
        https://medium.com/inside-machine-learning/what-is-a-transformer-d07dd1fbec04

        :param fasttext: bool. Si True, utilise fasttext
        """
        super().__init__(name="mlpTransformerFasttext"
                         ,model = None
                         ,id_run = id_run
                         ,path_run = path_run
                         )
        self.name = "mlpTransformerFasttext"
        self.fasttext = fasttext
        self.dict_info = None
        self.nb_transformerblocks = nb_transformerblocks
        self.num_heads = num_heads
        self.transformer_ffdim = transformer_ffdim
        self.denses_sizes = denses_sizes
        
    def train_model(self, X_train, y_train):
        tf.compat.v1.disable_eager_execution() # MEMORY LEAK OTHERWISE 
        flatten = lambda l: [item for sublist in l for item in sublist]

            
        with open('output/dict_info.p', 'rb') as file:
            self.dict_info = pickle.load(file)
        n_features = self.dict_info["n_features"]   
        len_largest_token = self.dict_info['len_largest_token']
        vocab_size = self.dict_info['vocab_size']
        list_len_tokens_col = self.dict_info['list_len_tokens_col']
        
        list_embedd_matrix=None
        if self.fasttext :
            list_embedd_matrix = np.load("embedding_fasttext.npy")
        
        
        X_train, X_eval, y_train, y_eval = train_test_split(X_train
                                                            , y_train
                                                            , test_size = 0.3
                                                            , random_state = 42
                                                            , stratify = y_train)
        
        
        ########
        # Model
        ########
        y_ints = [y.argmax() for y in y_train]
        class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_ints),
                                                 y_ints)
        dict_class_weight = dict(enumerate(class_weights))
        
        # plusieurs colonne -> plusieurs input
        list_input = []
        list_embedd = []

        my_model = TransformerModel(n_features,
                                    len_largest_token,
                                    vocab_size,
                                    list_len_tokens_col,
                                    nb_transformerblocks= self.nb_transformerblocks, 
                                    num_heads=self.num_heads, 
                                    transformer_ffdim=self.transformer_ffdim, 
                                    denses_sizes=self.denses_sizes,
                                    list_embedd_matrix=list_embedd_matrix)
        my_model.build((np.sum(list_len_tokens_col).item(),))
        my_model.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy())
        print(my_model.summary())
                
        mc = ModelCheckpoint('tmp/best_model', save_best_only=True, save_weights_only=True,  mode='auto',  verbose=1)
        es = EarlyStopping(monitor='val_loss', mode = 'min', verbose = 1, patience = 5)
        
        my_model.fit(np.asarray(X_train), y_train
                ,epochs = 1000
                ,batch_size = 32
                ,validation_data = (np.asarray(X_eval), y_eval)
#                 ,class_weight = dict_class_weight
                ,callbacks = [es, mc]
                ,verbose = True)
        
        self.model =  my_model
        self.model.load_weights('tmp/best_model')

        mydir = os.path.dirname(os.path.realpath(__file__))
        parent_path = os.path.dirname(os.path.normpath(mydir))
        my_model.save_weights(os.path.join(parent_path, "output", "model.h5"))

        filelist = []
#         for file in os.listdir("/home/jovyan/aiee2/pipeline/tmp"):
#             if file.endswith(".h5"):
#                 filelist.append(file)
                
#         for ind,f in enumerate(filelist):
#             print(f"-----{ind}  {f}------")
#             self.model.load_weights("tmp/"+f) 
            
#             self.model_name = f"{self.name}_eval_{f}"
#             predictions = self.model.predict(np.asarray(X_eval))
#             predictions = np.argmax(predictions,axis=1)
# #             pickle.dump( predictions, open( f"{f}_pred_eval.p", "wb" ) )
#             super().compute_metrics(y_eval[:,1], predictions)
            
#             self.model_name = f"{self.name}_train_{f}"
#             predictions = self.model.predict(np.asarray(X_train))
#             predictions = np.argmax(predictions,axis=1)
# #             pickle.dump( predictions, open( f"{f}_pred_train.p", "wb" ) )
#             super().compute_metrics(y_train[:,1], predictions)
        
        
#         pickle.dump( X_eval, open( f"X_eval.p", "wb" ) )
#         pickle.dump( X_train, open( f"X_train.p", "wb" ) )
        
       
#         pickle.dump( y_eval, open( f"y_eval.p", "wb" ) )
#         pickle.dump( y_train, open( f"y_train.p", "wb" ) )
# #         pickle.dump( X_train, open( f"X_train.p", "wb" ) )

    def load_model(self, model_path, dict_info_path='dict_info.pickle'):
        """
        Load function for runtime
        """
        if self.model is not None:
            raise RuntimeError('Trying to load already loaded model')

        with open(dict_info_path, 'rb') as file:
            self.dict_info = pickle.load(file)

        len_largest_token = self.dict_info['len_largest_token']
        vocab_size = self.dict_info['vocab_size']
        n_features = self.dict_info['n_features']
        list_len_tokens_col = self.dict_info['list_len_tokens_col']

        self.model = TransformerModel(n_features,
                                      len_largest_token,
                                      vocab_size,
                                      list_len_tokens_col,
                                      nb_transformerblocks= self.nb_transformerblocks, 
                                      num_heads=self.num_heads, 
                                      transformer_ffdim=self.transformer_ffdim, 
                                      denses_sizes=self.denses_sizes,
                                      list_embedd_matrix=np.zeros((vocab_size+1, n_features)))
        self.model.build((np.sum(list_len_tokens_col).item(),))
        self.model.load_weights(model_path)

    def predict(self,  X):
        if self.dict_info is None:
            raise RuntimeError('No model loaded')
        
        return self.model.predict(np.asarray(X))
        
    def run_model(self,  X_test,  y_test):
        
        mydir = os.path.dirname(os.path.realpath(__file__))
        mydir = os.path.dirname(os.path.normpath(mydir))
        
        filelist = []
        
        for file in os.listdir(os.path.join(mydir, "output")):
            if file.endswith(".h5"):
                filelist.append(file)
                
        for ind,f in enumerate(filelist):
            self.model.load_weights(f"{mydir}/output/"+f) 
            
            self.model_name = f"{self.name}_test_{f}"
            predictions = self.model.predict(np.asarray(X_test))
            predictions = np.argmax(predictions,axis=1)
#             pickle.dump( predictions, open( f"{f}_pred_test.p", "wb" ) )
            super().compute_metrics(y_test[:,1], predictions)
                
                
        pickle.dump( X_test, open( f"X_test.p", "wb" ) )
        pickle.dump( y_test, open( f"y_test.p", "wb" ) )        
        
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


if __name__ == "__main__":
    model = MLPTransformer(0, "")
    model.load_model("../output/best_model.h5", "../output")