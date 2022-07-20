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
from sklearn.model_selection import cross_val_score

import sklearn
from sklearn.svm import LinearSVC

from functools import partial

import json

from .Model import *

class Svm(Model):
    
    def __init__(self, id_run, path_run, tuning=False, hyper_param=None):
        """
        Modèle basé sur sklearn.svm.LinearSVC

        :param tuning: si True: utilise hyperopt pour optimiser une LinearSVC
        :param hyper_param: dict. si tuning=False, passé en hyperparametre à la LinearSVC
        """
        super().__init__(name="linearsvm"
                         ,model = None
                         ,id_run = id_run
                         ,path_run = path_run)
        self.tuning = tuning
        self.hyper_param = hyper_param
        
        
    def train_model(self, X_train,  y_train):
        if self.tuning is False:
            if self.hyper_param is None:
                self.model = LinearSVC()
                self.model.fit(X_train,y_train)
            else:
                self.model = LinearSVC(**self.hyper_param)
                self.model.fit(X_train,y_train)
        else:
            from hyperopt import fmin, tpe, hp
            space_svm = {
                'C':hp.loguniform('C',np.log(0.001), np.log(1000))
                    }
            fmin_objective = partial(self.objectiv_function_svm, X_train=X_train,y_train=y_train)
            
            best_param_svm = fmin(fmin_objective, space_svm, max_evals = 2, algo = tpe.suggest)
#             best_param_svm['class_weight'] ='balanced'
            best_param_svm['probability'] = True
#             weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(y_true),y=y_train)
#             best_param_svm['class_weight'] = weights
            self.hyper_param = best_param_svm
            my_svm = LinearSVC(**best_param_svm)
            my_svm.fit(X_train,y_train)
            self.model = my_svm
            
            with open(f'param_{self.model_name}_{self.id_run}.pickle', 'wb') as handle:
                pickle.dump(best_param_svm, handle, protocol=pickle.HIGHEST_PROTOCOL)
            save_file_on_minio(f"param_{self.model_name}_{self.id_run}.pickle", dir_name=self.path_run)
            
    def run_model(self,  X_test,  y_test):
        predictions = self.model.predict(X_test)
        super().compute_metrics(y_test, predictions)
    
    def objectiv_function_svm(self,space,X_train,y_train):
        C = float(space['C'])
#         weights = sklearn.utils.class_weight.compute_class_weight(class_weight='balanced',classes=np.unique(y_train),y=y_train.reshape(-1))
        clf = LinearSVC(C=C, probability=True)
        loss = cross_val_score(clf, X_train, y_train, cv=5, scoring='brier_score_loss',n_jobs = -1).mean()
        return loss