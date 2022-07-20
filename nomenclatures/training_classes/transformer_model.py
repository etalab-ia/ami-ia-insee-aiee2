"""
Implémentation du modèle transformer

Author : cpoulet@starclay.fr
date : 06/10/2020
"""

import os, sys
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

from .training_model import TrainingModel
from .similarity_model import SimilarityModel
from .token_field_and_position_embedding import TokenFieldAndPositionEmbedding


class MultiHeadSelfAttention(layers.Layer):
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
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1, **kwargs):
        """
        Block transformer

        :param embed_dim: taille d'entrée des inputs (taille des embeddings)
        :param num_head: nb de tête d'attention
        :param ff_dim: taille de la couche fully-connected interne
        :param dropout: dropout à appliquer
        """
        super(TransformerBlock, self).__init__(**kwargs)
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout)
        self.dropout2 = layers.Dropout(dropout)
        
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


class TransformerModel(TrainingModel):
    
    def __init__(self, path_run, 
                 nomenclature_distance, 
                 load_path = None,
                 seq_len=20, nb_fields=0, vocab_size=None,
                 embedding_size=256, 
                 pre_trained_embeddings_weights=None, pre_trained_weights_trainable=False,
                 nb_blocks=4, nb_heads=5, ff_dim=64,
                 dense_sizes=[256, 128, 64], dropout=0.1):
        """
        Modèle constitué de blocks Transformers

        :param seq_len: int. longueur max d'un champs
        :param nb_fields: int. nombre de champs
        :param vocab_size: int. taille du vocabulaire
        :param embedding_size: int. taille des embeddings
        :param pre_trained_embeddings_weights: np array. poids pré-entrainés à utiliser
        :param pre_trained_weights_trainable: bool, les poids pré-entrainés sont ils entrainables ?
        :param nb_blocks: int. nombre de blocks transformers
        :param nb_heads: int. nb de tête d'attention dans les blocks
        :param ff_dim: int. taille de la couche fully-connected interne aux blocks transformers
        :param dense_sizes: [int]. Taille des couches fully-connected en sortie
                            la dernière valeur est la taille de la projection
        :param dropout: dropout à appliquer
        """
        super().__init__('TransformerModel', path_run, nomenclature_distance, load_path,
                         seq_len=seq_len, nb_fields=nb_fields, vocab_size=vocab_size,
                         embedding_size=embedding_size,
                         pre_trained_embeddings_weights=pre_trained_embeddings_weights,
                         pre_trained_weights_trainable=pre_trained_weights_trainable,
                         nb_blocks=nb_blocks, nb_heads=nb_heads, ff_dim=ff_dim,
                         dense_sizes=dense_sizes, dropout=dropout)
        self.seq_len = seq_len
        self.nb_fields = nb_fields
        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        
    def define_layers(self, **kwargs):
        input_words = layers.Input(shape=(kwargs['seq_len']))
        input_fields = layers.Input(shape=(kwargs['seq_len']))
        input_positions = layers.Input(shape=(kwargs['seq_len']))
        
        embeddings = TokenFieldAndPositionEmbedding(seq_len=kwargs['seq_len'],
                                                    nb_fields=kwargs['nb_fields'],
                                                    vocab_size=kwargs['vocab_size'], 
                                                    embed_dim=kwargs['embedding_size'],
                                                    pre_trained_weights=kwargs['pre_trained_embeddings_weights'],
                                                    pre_trained_weights_trainable=kwargs['pre_trained_weights_trainable'],
                                                    name="embedding")
        x_with_pos = embeddings(input_words, input_fields, input_positions)
        for _ in range(kwargs['nb_blocks']):
            x_with_pos = TransformerBlock(embed_dim=kwargs['embedding_size'], 
                                          num_heads=kwargs['nb_heads'], 
                                          ff_dim=kwargs['ff_dim'],
                                          dropout=kwargs['dropout'])(x_with_pos)
        x_with_pos = layers.GlobalAveragePooling1D()(x_with_pos)
        for dense_size in kwargs['dense_sizes'][:-1]:
            x_with_pos = layers.Dense(dense_size, activation='relu')(x_with_pos)
            x_with_pos = layers.Dropout(kwargs['dropout'])(x_with_pos)
        output = layers.Dense(kwargs['dense_sizes'][-1], activation=None)(x_with_pos)
        output = output / tf.reshape(tf.norm(output, axis=-1), (-1, 1))

        return [input_words, input_fields, input_positions], output
