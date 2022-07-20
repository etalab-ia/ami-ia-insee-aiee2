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
import numpy as np 
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import calibration_curve
import sklearn

from .Model import *

class LogRegr(Model):
    
    def __init__(self, id_run, path_run):
        """
        Modèle basé sur sklearn.linear_model.LogisticRegression
        """
        super().__init__(name="Regression_logistique"
                         ,model = None
                         ,id_run = id_run
                         ,path_run = path_run)
        
    def train_model(self, X_train,  y_train):
#         weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(y_train),y=y_train.reshape(-1))
        self.model = LogisticRegression(n_jobs=-1)
        self.model.fit(X_train, y_train)
        
    def run_model(self,  X_test,  y_test):
        predictions = self.model.predict_proba(X_test)[:,1]
        # WARNING CUSTOM THRESHOLD
        predictions = predictions >= 0.5
        super().compute_metrics(y_test, predictions)
#         pred_proba = self.model.predict_proba(X_test)[:,1]
        
#         fop, mpv = calibration_curve(y_test, pred_proba, n_bins=10, normalize=True)
#         with open(f'fop.pickle', 'wb') as handle:
#                 pickle.dump(fop, handle, protocol=pickle.HIGHEST_PROTOCOL)
#         with open(f'mpv.pickle', 'wb') as handle:
#                 pickle.dump(mpv, handle, protocol=pickle.HIGHEST_PROTOCOL)
