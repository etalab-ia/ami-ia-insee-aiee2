from tensorflow import keras
from tensorflow.keras import layers

from .fasttext_singleton import FasttextSingleton


class TokenFieldAndPositionEmbedding(layers.Layer):
    def __init__(self, seq_len, nb_fields, vocab_size, embed_dim,
                 pre_trained_weights=None, pre_trained_weights_trainable=False,
                 **kwargs):
        """
        Embeddings conjugant texte des champs concaténés, indice du champ pour chaque token,
        et indice de la position du token dans son champ

        :param seq_len: longueur max d'un champs
        :param nb_fields: nombre de champs
        :param vocab_size: taille du vocabulaire
        :param embed_dim: taille des embeddings
        :param pre_trained_weights: poids pré-entrainés à utiliser
        :param pre_trained_weights_trainable: bool, les poids pré-entrainés sont ils entrainables ?
        """
        super(TokenFieldAndPositionEmbedding, self).__init__(**kwargs)
        if pre_trained_weights is None:
            self.token_emb = layers.Embedding(input_dim=vocab_size, 
                                              output_dim=embed_dim)
        
        else:
            self.token_emb = layers.Embedding(input_dim=vocab_size,
                                              output_dim=embed_dim,
                                              weights=[pre_trained_weights],
                                              trainable=pre_trained_weights_trainable)
        self.field_emb = layers.Embedding(input_dim=nb_fields, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=seq_len, output_dim=embed_dim)
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'token_emb': self.token_emb,
            'field_emb': self.field_emb,
            'pos_emb': self.pos_emb
        })
        return config

    def call(self, x, x_fields, x_positions):
        """
        Calcul des embeddings

        :returns: tf.vector
        """
        x = self.token_emb(x)
        fields = self.field_emb(x_fields)
        positions = self.pos_emb(x_positions)
        return x + fields + positions