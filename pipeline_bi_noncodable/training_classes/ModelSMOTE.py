"""
Classe de sur-échantillonage

https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html

Author : bsanchez@starclay.fr
date : 06/08/2020
"""

import pickle
from datetime import datetime
from . import utils
from .Model import *


class Smote(Model):
    
    def __init__(self, id_run, path_run):
        """
        Modèle basé sur imblearn.over_sampling.SMOTE
        """
        from imblearn.over_sampling import SMOTE
        super().__init__(name="smote"
                         ,model = SMOTE(sampling_strategy='auto')
                         ,id_run = id_run
                         ,path_run = path_run)
        
    def modify(self, X_train, y_train):
        X_train, y_train = self.model.fit_resample(X_train, y_train)
        return X_train, y_train
        
    def train_model(self,  X_test,  y_test):
        X_train, y_train = self.model.fit_resample(X_train, y_train)
        return X_train, y_train
    
    def run_model(self, X_train, X_test, y_train, y_test):
        ret