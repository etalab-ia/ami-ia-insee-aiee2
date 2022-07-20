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
from sklearn.model_selection import StratifiedKFold
from functools import partial

import xgboost as xgb


import json

from .Model import *

class XGBoost(Model):
    
    def __init__(self, id_run, path_run, tuning=False, hyper_param=None):
        """
        Modèle basé sur xgboost.xgb

        :param tuning: si True: utilise hyperopt pour optimiser un xgb
        :param hyper_param: chemin vers un fichier. si tuning=False, passé en hyperparametre au xgb
        """
        super().__init__(name="xgboost"
                         ,model = None
                         ,id_run = id_run
                         ,path_run = path_run)
        self.tuning = tuning
        self.hyper_param = hyper_param
        
        
    def train_model(self, X_train,  y_train):
        if self.tuning is False:
            if self.hyper_param is None:
                param = {'objective':'binary:hinge'}
                dtrain = xgb.DMatrix(X_train, label=y_train,nthread=-1)
                self.model = xgb.train(param, dtrain)
            else:
                with open(self.hyper_param, 'rb') as handle:
                    dict_hyper = pickle.load(handle)
                dtrain = xgb.DMatrix(X_train, label=y_train,nthread=-1)
                self.model = xgb.train(dict_hyper, dtrain)
        else:
            from hyperopt import fmin, tpe, hp

            space_xgb = {
                'n_estimators': hp.quniform('n_estimators', 2, 600, 1),
                'eta': hp.uniform('eta', 0.0001, 3), 
                'max_depth':  hp.choice('max_depth', np.arange(1, 14, dtype=int)),
                'min_child_weight': hp.quniform('min_child_weight', 1, 10, 1),
                'subsample': hp.quniform('subsample', 0.7, 1, 0.05),
                'gamma': hp.quniform('gamma', 0.5, 1, 0.05),
                'colsample_bytree': hp.quniform('colsample_bytree', 0.7, 1, 0.05),
                'alpha' :  hp.quniform('alpha', 0, 10, 1),
                'lambda': hp.quniform('lambda', 1, 2, 0.1)
            }
            
            fmin_objective = partial(self.objectiv_function_xgb, X_train=X_train,y_train=y_train )
            best_param_xgb = fmin(fmin_objective, space_xgb, max_evals = 2, algo = tpe.suggest)
            
            best_param_xgb['objective'] = 'binary:hinge'
            
            self.hyper_param = best_param_xgb
            dtrain = xgb.DMatrix(X_train, label=y_train,nthread=-1)
            self.model = xgb.train(best_param_xgb, dtrain,)
            with open(f'param_{self.model_name}_{self.id_run}.pickle', 'wb') as handle:
                pickle.dump(best_param_xgb, handle, protocol=pickle.HIGHEST_PROTOCOL)
            save_file_on_minio(f"param_{self.model_name}_{self.id_run}.pickle", dir_name=self.path_run)

            
    def run_model(self,  X_test,  y_test):
        dtest = xgb.DMatrix(X_test, label=y_test,nthread=-1)
        predictions = self.model.predict(dtest)
        super().compute_metrics( y_test, predictions)
    
    def objectiv_function_xgb(self,space,X_train,y_train):
        
        n_estimators = int(space['n_estimators'])
        eta = space['eta']
        max_depth = int(space['max_depth'])
        min_child_weight = int(space['min_child_weight'])
        subsample = space['subsample']
        gamma = space['gamma']
        colsample_bytree = space['colsample_bytree']
        alpha = space['alpha']
        lmbda =  space['lambda']
        
        
        scores = []
        kf = StratifiedKFold(n_splits=5)

        for train_index, test_index in kf.split(X_train, y_train):
            Xcv_train, Xcv_test = X_train[train_index],X_train[test_index]
            ycv_train, ycv_test = y_train[train_index],y_train[test_index]
            
            
            model = xgb.XGBClassifier(n_estimators=n_estimators
                                     ,eta=eta
                                     ,max_depth = max_depth
                                     ,min_child_weight = min_child_weight
                                     ,subsample = subsample
                                     ,gamma = gamma
                                     ,colsample_bytree = colsample_bytree
                                     ,alpha = alpha
                                     ,reg_lambda  = lmbda)
            
            model.fit(Xcv_train, ycv_train)
            
            predictions = model.predict_proba(Xcv_test)[:,1]
            
            scores.append(brier_score_loss(ycv_test, predictions))
                
        loss = np.mean(scores)   
            
        return loss