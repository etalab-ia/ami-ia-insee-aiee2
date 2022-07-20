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

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support as score
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
from autosklearn.experimental.askl2 import AutoSklearn2Classifier
from sklearn.metrics import balanced_accuracy_score

from .Model import *

class AutoSK(Model):
    
    def __init__(self, id_run, path_run):
        """
        Modèle basé sur autosklearn.classification.AutoSklearnClassifier
        """
        super().__init__(name="AutoSklearn"
                         ,model = autosklearn.classification.AutoSklearnClassifier(metrics="balanced_accuracy")
                         ,id_run = id_run
                         ,path_run = path_run)
        
    def train_model(self, X_train,  y_train):
        self.model.fit(X_train, y_train)
        
    def run_model(self,  X_test,  y_test):
        predictions = self.model.predict(X_test)
        super().compute_metrics(y_test, predictions)