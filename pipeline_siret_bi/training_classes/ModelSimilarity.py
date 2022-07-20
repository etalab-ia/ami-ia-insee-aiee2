"""
Modèle de similarité :
On entraine un embedding personalisé qui permet de plonger les BI et les Sirus dans un même espaces de représentation.
Afin de matcher un bi, on regarde les sirus les plus simulaires (cosine similarity).

Author : bsanchez@starclay.fr
date : 23/09/2020
"""
import os
import tempfile
import pickle
from tqdm import tqdm
import glob

import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model

import fasttext
import fasttext.util

from .Model import Model
from .utils import *
from .anchor_positive_pairs import AnchorPositivePairs
from .similarity_model import SimilarityModel


def uniquify(path: str):
    """
    Genère un nom de fichier unique
    Ex:
    mon_dossier/
        nouveau_fichier.txt
        nouveau_fichier_1.txt  <-- nom_générer par cette fonction
        
    :param: path: chemin du dossier
    :returns: str nom fichier unique
    """
    filename, extension = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = filename + "_" + str(counter)  + extension
        counter += 1
    return path


#####################
#  Modèle Transformer
#####################

class MultiHeadSelfAttention(layers.Layer):
    """
    class transformer, inspiré de https://keras.io/examples/nlp/text_classification_with_transformer/
    
    """
    def __init__(self, embed_dim, num_heads=8,**kwargs):
        """
        couche d'attention

        :param embed_dim: taille d'entrée des inputs (taille des embeddings)
        :param num_head: nb de tête d'attention
        """
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
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  
        key = self.key_dense(inputs)  
        value = self.value_dense(inputs)  
        query = self.separate_heads(query, batch_size) 
        key = self.separate_heads(key, batch_size)  
        value = self.separate_heads(value, batch_size)  
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])  
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))  
        output = self.combine_heads(concat_attention)  
        return output
    

class TransformerBlock(layers.Layer):
    """
    class transformer, inspiré de https://keras.io/examples/nlp/text_classification_with_transformer/
    
    """
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1, **kwargs):
        """
        Block transformer

        :param embed_dim: taille d'entrée des inputs (taille des embeddings)
        :param num_head: nb de tête d'attention
        :param ff_dim: taille de la couche fully-connected interne
        :param rate: dropout à appliquer
        """
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
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


class TokenFieldAndPositionEmbedding(layers.Layer):
    def __init__(self, maxseqlen, vocab_size, embed_dim, 
                 pre_trained_weights=None, pre_trained_trainable=True, **kwargs):
        """
        Embeddings conjugant texte des champs concaténés, indice du champ pour chaque token,
        et indice de la position du token dans son champ

        :param maxseqlen: longueur max d'un champs
        :param vocab_size: taille du vocabulaire
        :param embed_dim: taille des embeddings
        :param pre_trained_weights: poids pré-entrainés à utiliser
        :param pre_trained_trainable: poids pré-entrainés sont ils entrainables ?
        """
        super(TokenFieldAndPositionEmbedding, self).__init__(**kwargs)
        if pre_trained_weights is None:
            self.token_emb = layers.Embedding(input_dim=vocab_size
                                              , output_dim=embed_dim
                                              , weights = pre_trained_weights
                                              , trainable = pre_trained_trainable
                                              , mask_zero = True)
        else:
            self.token_emb = layers.Embedding(input_dim=vocab_size
                                              , output_dim=embed_dim
                                              , mask_zero = True)
            
        self.field_emb = layers.Embedding(input_dim = 69, output_dim=embed_dim) 
        self.pos_emb = layers.Embedding(input_dim=maxseqlen, output_dim=embed_dim)
        
    def get_config(self):
        config = super(TokenFieldAndPositionEmbedding, self).get_config()
        config.update({
            'token_emb': self.token_emb,
            'field_emb': self.field_emb,
            'pos_emb': self.pos_emb
        })
        return config

    def call(self, x, x_fields, x_positions):
        x = self.token_emb(x)
        fields = self.field_emb(x_fields)
        positions = self.pos_emb(x_positions)
        return x + fields + positions

 
class MLPSimilarity(Model):
    
    def __init__(self, 
                 local_path_run, 
                 fasttext = False, 
                 embedding_size = 120, 
                 nb_blocks=1, nb_heads=3, ff_dim=64,
                 denses_sizes=[128]):
        """
        Modèle transformer lui-même
        :param local_path_run: chemin du training en cours
        :param fasttext: bool. Si True, on charge les embeddings sauvegardé par le ProcessMLP
        :param embedding_size: int, taille des embeddings
        :param nb_blocks: int. nombre de blocks transformers
        :param nb_heads: int. nb de tête d'attention dans les blocks
        :param ff_dim: int. taille de la couche fully-connected interne aux blocks transformers
        :param dense_sizes: [int]. Taille des couches fully-connected en sortie
                            la dernière valeur est la taille de la projection
        """
        super().__init__(name="mlp_similarity", model=None, local_path_run=local_path_run)
        self.model = None
        self.dict_info = None
        self.fasttext = fasttext
        self.embedding_size = embedding_size
        self.nb_blocks = nb_blocks
        self.nb_heads = nb_heads
        self.ff_dim = ff_dim
        self.denses_sizes = denses_sizes
        if self.embedding_size % self.nb_heads != 0:
            raise ValueError('embedding_size must be divisible by nb_head')

    def define_model(self, seq_len, vocab_size):
        """
        Définition du modèle

        :param seq_len: int. longueur max d'un champs
        :param vocab_size: int. taille du vocabulaire
        :returns: [inputs], output
        """
        # Declaration modèle                    
        input_word_embeddings = layers.Input(shape=(seq_len))
        input_fields_embeddings = layers.Input(shape=(seq_len))
        input_positions_embeddings = layers.Input(shape=(seq_len))
        
        if self.fasttext:
            embedd_matrix = np.load(os.path.join(self.local_path_run, "embedding_fasttext.npy"))
            embeddings = TokenFieldAndPositionEmbedding(maxseqlen=seq_len,
                                                        vocab_size=vocab_size, 
                                                        embed_dim=self.embedding_size, 
                                                        name="fasttext_embedding", 
                                                        pre_trained_weights=[embedd_matrix],
                                                        pre_trained_trainable=True)
        else:
            embeddings = TokenFieldAndPositionEmbedding(maxseqlen = seq_len,
                                                        vocab_size = vocab_size, 
                                                        embed_dim=self.embedding_size, 
                                                        name = "embedding")

        x_with_pos = embeddings(input_word_embeddings, input_fields_embeddings, input_positions_embeddings)
        x_with_pos = TransformerBlock(self.embedding_size, self.nb_heads, self.ff_dim)(x_with_pos)
        for _ in range(self.nb_blocks):
            x_trans = TransformerBlock(self.embedding_size, 
                                       self.nb_heads, 
                                       self.ff_dim)(x_with_pos)
            x_with_pos = layers.LayerNormalization(epsilon=1e-6)(x_trans + x_with_pos)
        x_with_pos = layers.GlobalAveragePooling1D()(x_with_pos)
        for dense_size in self.denses_sizes[:-1]:
            x_with_pos = layers.Dense(dense_size, activation='relu')(x_with_pos)
            x_with_pos = layers.Dropout(0.1)(x_with_pos)
        output = layers.Dense(self.denses_sizes[-1], activation=None)(x_with_pos)
        output = output / tf.reshape(tf.norm(output, axis=-1), (-1, 1))
        return [input_word_embeddings, input_fields_embeddings, input_positions_embeddings], output
        
    def train_model(self, X_train, y_train, labels_train, 
                          X_val, y_val, labels_val, 
                          batch_size=32, nb_epochs=10):
        """
        Training

        :param X_train: données input BI train
        :param y_train: données input SI train
        :param labels_train: labels train (liste de SIRET)
        :param X_val: données input BI eval
        :param y_val: données input BI eval
        :param labels_val: labels eval (liste de SIRET)
        :param batch_size: taille de batch
        :param nb_epochs: nb d'époques
        """

        ########################################
        ########## Model #######################
        ########################################
        tf.compat.v1.enable_eager_execution() # S'assurer qu'on est en eager mode, autrement keras plante
        
        with open(os.path.join(self.local_path_run, 'dict_info.pickle'), 'rb') as file:
            self.dict_info = pickle.load(file)
        n_features : int = self.dict_info["n_features"]   
        seq_len : int = self.dict_info['len_full_largest_token']
        vocab_size_full : int = self.dict_info['vocab_size_full']

        inputs, output = self.define_model(seq_len, vocab_size_full)
        self.model = SimilarityModel(inputs=inputs, outputs=output)
        print(self.model.summary())
        
        # Compilation modèle
        self.model.compile(optimizer='adam', loss = keras.losses.MeanSquaredError(), run_eagerly=True)
        self.model.save(os.path.join(self.local_path_run, 'model_start'), save_format='tf')

        #Training
        train_data = AnchorPositivePairs(num_batchs=len(X_train) // batch_size, 
                                         input_data=X_train, 
                                         batch_size=batch_size, 
                                         pos_data=y_train, 
                                         labels=labels_train, 
                                         seq_len=seq_len)
        val_data = AnchorPositivePairs(num_batchs=len(X_val) // batch_size, 
                                       input_data=X_val, 
                                       batch_size=batch_size, 
                                       pos_data=y_val, 
                                       labels=labels_val, 
                                       seq_len=seq_len)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)
        os.makedirs(os.path.join(self.local_path_run, 'model_training'), exist_ok=True)
        mc = ModelCheckpoint(os.path.join(self.local_path_run, 'model_training', 'best_model_{epoch:02d}-{val_loss:.5f}.h5'), 
                             monitor='val_loss', mode='min', verbose=1, save_weights_only=False, save_best_only=True)
        self.model.fit(train_data, validation_data=val_data, epochs=nb_epochs, verbose=True, callbacks=[es, mc])
        
        best_save_model = max(glob.glob(os.path.join(self.local_path_run, 'model_training', '*')), key=os.path.getctime)
        self.model.load_weights(best_save_model)
        
########################################################################################################################################
    def save_model(self, save_dir):
        """
        sauvegarde du modèle

        :param save_dir: dossier ou sauver
        """
        if save_dir is None:
            save_dir = self.local_path_run
        self.model.save(save_dir,save_format='tf')

    def load_model(self, model_path, dict_info_path):
        """
        Load function for runtime

        :param model_path: chemin vers la sauvegarde du modèle
        :param dict_info_path: chemin vers le dossier contenant le fichier de sauvegarde des meta-paramètres
        """
        if self.model is not None:
            raise RuntimeError('Trying to load already loaded model')

        self.model = keras.models.load_model(model_path)
    
        with open(os.path.join(dict_info_path, 'dict_info.pickle'), 'rb') as file:
            self.dict_info = pickle.load(file)
        
    def predict(self, X, path_output):
        """
        Prédiction d'un modèle déjà entrainée

        :param X: data à prédire (BI pré-traitée mais pas batchée via AnchorPositivePairs)
        :param path_output: dossier local ou sauvegarder les prédictions
        """
        if self.model is None:
            raise RuntimeError('Model not loaded')

        n_features = self.dict_info["n_features"]   
        len_full_largest_token = self.dict_info['len_full_largest_token']
        vocab_size_full = self.dict_info['vocab_size_full']
        
        print(f"len_full_largest_token is {len_full_largest_token}")
        
        matrix_process = np.empty([len(X), 3, len_full_largest_token]) # Longueur du nb d'elem à prédire, 3 type d'embedding, longueur plus long token
        
        idx = 0
        for x, y in tqdm(zip(X, X)):
            dummy = [[666,666]] # Les labels utilisé dans la matrice d'entrainement ne sont pas utilisé,
                                # mais on a besoin d'une valeur fictive pour AnchorPositivePairs
            item = next(iter(AnchorPositivePairs(num_batchs=1, 
                                                 batch_size=1, 
                                                 input_data=np.array([x]), 
                                                 pos_data=np.array([x]), 
                                                 labels=dummy, 
                                                 seq_len=len_full_largest_token)))
            matrix_process[idx,0] = np.array([item[0,0]], dtype=np.float32)
            matrix_process[idx,1] = np.array([item[1,0]], dtype=np.float32)
            matrix_process[idx,2] = np.array([item[2,0]], dtype=np.float32)
            idx += 1
            
        chunk_size = 10000 # valeur arbitraire
        
        part = 0
        for i in range(chunk_size, len(X) + chunk_size, chunk_size):
            j = i - chunk_size
            if(i > len(X)):
                j = i - chunk_size
                i = len(X)
           
            matrix_embedding = np.empty([len(X), 1]) # La taille s'adeptera sur la taille d'embedding
            matrix_embedding = self.model.predict([matrix_process[:,0][j:i], matrix_process[:,1][j:i], matrix_process[:,2][j:i]])
            nname = uniquify(os.path.join(path_output, "matrix_embedding.p"))
            with open(nname, "wb") as output_file:
                pickle.dump(matrix_embedding, output_file)
            part = part + 1
        
    def run_model(self, X_test, y_test):
        pass
    
