"""
Classe mère des classe comportant des modèles

Elle gere la sauvegarde automatique des modèle
Author : bsanchez@starclay.fr
date : 06/08/2020
"""
import os
from abc import ABC
from abc import ABCMeta
from abc import abstractmethod
from abc import ABC
import tempfile
import pickle
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import brier_score_loss
from sklearn.metrics import log_loss
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import fbeta_score
from sklearn.metrics import  precision_score
from sklearn.metrics import  recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score

from .utils import *


class Model(ABC):
    
    def __init__(self, name, model, local_path_run, model_load_path=None, dict_info_path=None):
        """
        Classe mère des classe comportant des modèles
        Elle gere la sauvegarde automatique des modèle

        :param name: str. Nom du modèle
        :param model: modèle de prédiction (keras ou autre) chargé
        :param local_path_run: chemin local du training
        :param model_load_path: chemin local pour charger le modèle entraîné
        :param dict_info_path: chemin local vers le fichier contenant les meta-parametres
        """
        self.model_name = name
        self.model = model
        if model_load_path is not None:
            self.load_model(model_load_path, dict_info_path)
        self.local_path_run = local_path_run
                    
    @abstractmethod
    def train_model(self, X_train,  y_train):
        """
        Method qui entraine le modèle, avec ou sasn tuning
        
        :param X_train: matrice creuse contenant les donnée d'entrainement
        :param y_train: Matrice pleine contenant les labels
    
        :returns: void
        """
        pass
    
    @abstractmethod
    def run_model(self,  X_test,  y_test):
        """
        Method qui prédit les label sur des données fournis.
        
        :param X_test: matrice creuse contenant les donnée de test
        :param y_test: Matrice pleine contenant les label
    
        :returns: void
        """
        pass
        
    def save_model(self, save_dir=None):
        """
        Sauvegarde le modèle
        :returns: void
        """
        if save_dir is None:
            save_dir = self.local_path_run
        pickle.dump(self.model, open(os.path.join(save_dir, self.model_name + ".p"), "wb"))
    
    def load_model(self, model_path, dict_info_path):
        """
        Load function for runtime

        :param model_path: chemin vers la sauvegarde du modèle
        :param dict_info_path: chemin vers le fichier de sauvegarde des meta-paramètres
        :returns: None
        """
        self.model = pickle.load(open(model_path, 'rb'))
        
    def compute_metrics(self, y_true, prediction):
        """
        Génère les métrique de performance et les sauvegarde
        
        :param prediction: Matrice dense contenant les labels
        :param y_test: Matrice dense contenant les labels
        :returns: void
        """
        # Threshold ?
        dict_metric = {}
        cm = confusion_matrix(y_true, prediction)
        tn, fp, fn, tp = cm.ravel()
        dict_metric['model'] = self.model_name
        dict_metric['tn'] = tn
        dict_metric['fp'] = fp
        dict_metric['fn'] = fn
        dict_metric['tp'] = tp
        
        dict_metric['tn_rate'] = tn / (tn + fp)
        dict_metric['fp_rate'] = fp / (fp + tn)
        dict_metric['fn_rate'] = fn / (fn + tp)
        dict_metric['tp_rate'] = tp / (tp + fn)
        
        dict_metric['false_positive_rate'] = false_positive_rate = fp / (fp + tn)
        dict_metric['false_negative_rate'] = false_negative_rate = fn / (tp + fn)
        dict_metric['true_negative_rate'] = true_negative_rate = tn / (tn + fp)
        dict_metric['negative_predictive_value'] = negative_predictive_value = tn/ (tn + fn)
        dict_metric['false_discovery_rate'] = false_discovery_rate = fp/ (tp + fp)
        dict_metric['true_positive_rate'] = true_positive_rate = tp / (tp + fn)
        dict_metric['positive_predictive_value'] = positive_predictive_value = tp/ (tp + fp)
        dict_metric['accuracy'] =  (tp + tn) / (tp + tn + fp + fn)
        dict_metric['roc_auc'] = roc_auc_score(y_true, prediction)
        dict_metric['average_precision_score'] = average_precision_score(y_true, prediction)
        dict_metric['log_loss'] = log_loss(y_true, prediction)
        dict_metric['brier_score_loss'] = brier_score_loss(y_true, prediction)
        dict_metric['fbeta_0.5'] = fbeta_score(y_true, prediction, beta=0.5)
        dict_metric['fbeta_2'] = fbeta_score(y_true, prediction, beta=2)
        dict_metric['f1_score'] = f1_score(y_true, prediction)

        #         fbeta_score(y_true, prediction, beta)
        #         cohen_kappa_score(y_true, prediction)
        
        df = pd.DataFrame([dict_metric])
        return df