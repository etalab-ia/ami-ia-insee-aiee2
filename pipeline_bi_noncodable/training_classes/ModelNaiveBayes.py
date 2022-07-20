"""
Classe de modèle

Author : bsanchez@starclay.fr
date : 06/08/2020
"""

import gensim
import tempfile
import pickle
from datetime import datetime
from . import utils

from sklearn.naive_bayes import MultinomialNB

from .Model import *

class NaiveBaye(Model):
    
    def __init__(self, id_run, path_run):
        """
        Modèle basé sur sklearn.naive_bayes.MultinomialNB
        """
        super().__init__(name="NaiveBaye"
                         ,model = MultinomialNB()
                         ,id_run = id_run
                         ,path_run = path_run)
        
    def train_model(self, X_train,  y_train):
        self.model.fit(X_train, y_train)

    def run_model(self,  X_test,  y_test):
        predictions = self.model.predict(X_test)
        super().compute_metrics(y_test, predictions)
        #predict proba useless on NaiveBaye : https://scikit-learn.org/stable/modules/naive_bayes.html
