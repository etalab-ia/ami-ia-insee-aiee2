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
from sklearn.linear_model import LogisticRegressionCV
from .Model import *

class LogRegrCV(Model):
    
    def __init__(self, id_run, path_run):
        """
        Modèle basé sur sklearn.linear_model.LogisticRegressionCV
        """
        super().__init__(name="Regression_logistiqueCV"                         
                         ,model = None
                         ,id_run = id_run
                         ,path_run = path_run)
        
        
    def train_model(self, X_train,  y_train):
#         weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(y_train),y=y_train)
        self.model = LogisticRegressionCV(cv=20, random_state=42, scoring='brier_score_loss',max_iter=1000, n_jobs = -1)
        self.model.fit(X_train, y_train)
            
    def run_model(self,  X_test,  y_test):
        predictions = self.model.predict(X_test)
        super().compute_metrics(y_test, predictions)
        self.model.predict_proba(X_test)
