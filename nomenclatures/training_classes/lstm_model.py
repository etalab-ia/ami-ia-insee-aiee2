import os, sys
from tensorflow import keras
from tensorflow.keras import layers

from .training_model import TrainingModel
from .similarity_model import SimilarityModel
from .token_field_and_position_embedding import TokenFieldAndPositionEmbedding


class LstmModel(TrainingModel):
    
    def __init__(self, path_run, 
                 nomenclature_distance, 
                 load_path = None,
                 seq_len=20, nb_fields=0, vocab_size=None,
                 embedding_size=256):
        """
        Modèle constitué de couches de LSTM
        """
        super().__init__('LstmModel', path_run, nomenclature_distance, load_path,
                         seq_len=seq_len, nb_fields=nb_fields, vocab_size=vocab_size,
                         embedding_size=embedding_size)
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
                                                    name="embedding")
        x_with_pos = embeddings.call(x=input_words, x_fields=input_fields, x_positions=input_positions)
        x_with_pos = layers.Bidirectional(layers.LSTM(int(kwargs['seq_len']), return_sequences=True))(x_with_pos)
        x_with_pos = layers.Dropout(0.1)(x_with_pos)
        x_with_pos = layers.Bidirectional(layers.LSTM(int(kwargs['seq_len']), return_sequences=True))(x_with_pos)
        x_with_pos = layers.Dropout(0.1)(x_with_pos)
        x_with_pos = layers.Dense(256, activation='relu')(x_with_pos)
        x_with_pos = layers.Dropout(0.1)(x_with_pos)
        x_with_pos = layers.GlobalAveragePooling1D()(x_with_pos)
        x_with_pos = layers.Dense(128, activation='relu')(x_with_pos)
        x_with_pos = layers.Dropout(0.1)(x_with_pos)
        output = layers.Dense(64, activation=None)(x_with_pos)
        
        return [input_words, input_fields, input_positions], output
