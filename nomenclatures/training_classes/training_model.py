"""
Classe mère des classe comportant des modèles

Elle gere la sauvegarde automatique des modèle
Author : bsanchez@starclay.fr
date : 06/08/2020
"""

from abc import ABC
from abc import ABCMeta
from abc import abstractmethod
from abc import ABC
import tempfile
import pickle
import logging
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

import os, sys
from .similarity_model import SimilarityModel


class TrainingModel(ABC):
    
    def __init__(self, name, path_run,
                 nomenclature_distance, 
                 load_path=None, **kwargs):
        """
        Superclasse pour les modèles à entrainer

        :param name: nom du modèle
        :param path_run: chemin de sauvegarde
        :param nomenclature_distance: NomenclatureDistance à utiliser
        :param load_path: chemin à charger
        :param kwargs: dict d'arguments pour la construction du modèle (passé à build_model, puis define_layers)
        """
        self.model_name = name
        self.path_run = path_run
        self.logger = logging.getLogger(name)
        
        self.nomenclature_distance = nomenclature_distance
        if load_path is not None:
            self.model = self.load_model(load_path)
        else:
            self.model = self.build_model(**kwargs)
        self.logger.info(self.model.summary())
        
    @abstractmethod
    def define_layers(self, **kwargs):
        """
        Methode qui définit les couches
        
        :params kwargs: dict d'arguments pour la construction des couches
    
        :returns: input (tf.Tensor ou list de tf.tensor), 
                  output (tf.Tensor ou list de tf.tensor)
        """
        raise NotImplementedError()
    
    def build_model(self, **kwargs):
        """
        Method qui crée les couches via define_layers, puis encapsule dans un SimilarityModel
        
        :params kwargs: dict d'arguments pour la construction des couches
        :returns: keras.Model
        """
        inputs, output = self.define_layers(**kwargs)

        model = SimilarityModel(inputs=inputs, 
                                outputs=output, 
                                nomenclature_distance=self.nomenclature_distance)
        return model
                    
    def train_model(self, training_pairs_batcher, validation_pairs_batcher, nb_epochs):
        """
        Entraine le modèle:
            - compile le modèle
            - ajoute des callbacks (ModelCheckpoint, EarlyStopping)
            - lance le training

        :param training_pairs_batcher: AnchorPositivePairsBatch contenant les données de training
        :param validation_pairs_batcher: AnchorPositivePairsBatch contenant les données de validation
        :param nb_epochs: nb d'époques de training
        :returns: training History
        """
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=keras.losses.MeanSquaredError(),  
#             run_eagerly=tf.executing_eagerly()
        )
        self.logger.info(f'Running eagerly : {tf.executing_eagerly()}')
        
        train_save_path = os.path.join(self.path_run, 'train_weights')
        mc = ModelCheckpoint(os.path.join(train_save_path, 'best_model_{epoch:02d}-{val_loss:.5f}'), 
                             save_best_only=True, save_weights_only=False,  
                             mode='auto', period=1, verbose=1)
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
        if not os.path.exists(train_save_path):
            os.makedirs(train_save_path)

        self.logger.info('training')
        history = self.model.fit(training_pairs_batcher, 
                                 epochs=nb_epochs,
                                 validation_data=validation_pairs_batcher,
                                 callbacks = [es, mc],
                                 verbose = True)
        return history

    def run_model_siamese(self,  testing_pairs_batcher):
        return
    
    def run_model_single_side(self,  formatted_data):
        """
        prediction pour formatted_data (formatté via AnchorPositivePairsBatch.format_input)

        :param formatted_data: tf.tensor
        :returns: tf.tensor
        """
        return self.model(formatted_data, training=False)
    
    def load_model(self,path):
        """
        Charge un modele
        IMPORTANT : le modèle chargé est de type keras.Model, pas SimilarityModel
        
        :params path: chemin du model à être charger
        :returns: void
        """
        return keras.models.load_model(path)
