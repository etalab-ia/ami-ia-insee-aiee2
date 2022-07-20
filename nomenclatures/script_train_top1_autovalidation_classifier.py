#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# train_top1_autovalidation_classifier.py [training_dir]
#     training_dir : dossier du training. 
#                    Défault : training le plus récent dans config['local']['trainings_dir']/config['data']['nomenclature']['name']
#
# script qui entraine un classifier à décider si la valeur prédite en top-1 est à coder automatiquement,
# en se basant sur des features calculées sur les scores des top-1 et top-2 prédits
#
# Pour cela, de nombreux classifieurs sont testés, et le meilleur est sauvegardé.
#
# Pour choisir le meilleur, on optimise un critère parmi 'precision', 'recall' et f1,
# paramêtré dans config['top1_classifier']['metric_to_maximize']:
# Pour chaque modèle:
#    - on l'entraîne
#    - pour chaque valeur possible de seuil, on calcule précision/recall/f1
#    - à la fin, on garde la meilleure combinaison modèle/seuil
#
# La classe Top1ValidatorModel est mise à disposition pour charger simplement le modèle et l'utiliser.
#
# utilise config.yaml dans le dossier de training
#
# @author cyril.poulet@starclay.fr
# @date: nov 2020
import logging.config
import s3fs
import pandas as pd
import json, yaml, pickle
import os
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

from training_utils import save_config, load_config, push_to_minio, get_trainings_dir, get_last_local_training_dir
    
    
def calculate_metrics(clf, X, y, threshold):
    """
    Calculates precision, recall and f1 for the given parameters

    :param clf: sklearn trained classifier
    :param X: formatted input (dataframe)
    :param y: labels
    :param threshold: float, threshold to use. only scores for class 1 over threshold are validated
    :returns: precision, recall and f1 for given threshold
    """
    predictions = pd.DataFrame()
    predictions['scores'] = clf.predict_proba(X).tolist()
    predictions['preds'] = predictions['scores'].map(lambda x: x[1] >= threshold)
    predictions['tp'] = (predictions['preds'] == 1) & (y == 1)
    predictions['tn'] = (predictions['preds'] == 0) & (y == 0)
    predictions['fp'] = (predictions['preds'] == 1) & (y == 0)
    predictions['fn'] = (predictions['preds'] == 0) & (y == 1)
    precision = sum(predictions['tp']) / max(1, (sum(predictions['tp']) + sum(predictions['fp'])))
    recall = sum(predictions['tp']) / max(1, (sum(predictions['tp']) + sum(predictions['fn'])))
    f1 = 2 * (precision*recall)/max(1, (precision+recall))
    return precision, recall, f1
            
    
class Top1ValidatorModel:
    
    def __init__(self, save_dir):
        """
        Simple class loading classifier, formatting input and applying threshold.
        It is meant to validate when the model is sure that the top-1 class is correct.

        :param save_dir: directory containing the model
        """
        config = load_config(save_dir)
        with open(os.path.join(save_dir, config['top1_classifier']['scaler_file']), 'rb') as f:
            self.scaler = pickle.load(f)
        with open(os.path.join(save_dir, config['top1_classifier']['model_file']), 'rb') as f:
            self.clf = pickle.load(f)
        self.threshold = config['top1_classifier']['threshold']
        
    def validate_top1(self, top_k_scores):
        """
        calculates input, predict if top-1 is correct (if score for class 1 is over threshold)

        :param top_k_scores: np.array of size N*k (batched top-k scores)
        :returns: np array of booleans, size N
        """
        data = pd.DataFrame()
        data['top_k_similarities'] = top_k_scores
        data['top_1_score'] = data['top_k_similarities'].map(lambda x: x[0])
        data['top_2_score'] = data['top_k_similarities'].map(lambda x: x[1])
        data['top_12_diff'] = data['top_k_similarities'].map(lambda x: x[0] - x[1])
        data['top_12_normdiff'] = data['top_k_similarities'].map(lambda x: (x[0] - x[1])/x[0])
        data['top_12_div'] = data['top_k_similarities'].map(lambda x: x[0] / x[1])
        data = data.drop(columns=['top_k_similarities'])
        data = self.scaler.transform(data)

        res = self.clf.predict_proba(data)
        return res[:, 1] >= self.threshold
    
    
if __name__ == "__main__":

    """
    Script permettant de calculer les performance d'un training en top-k

    """

    # Logging - Chargement du fichier de configuration log
    with open(os.path.join(os.path.dirname(__file__), 'logging.conf.yaml'), 'r') as stream:
        log_config = yaml.load(stream, Loader=yaml.FullLoader)
    logging.config.dictConfig(log_config)

    logger = logging.getLogger()

    # path to test
    with open("config.yaml") as f:
        base_config = yaml.safe_load(f)
    TRAININGS_LOCAL_DIR, _ = get_trainings_dir(base_config)

    parser = argparse.ArgumentParser(
        description="Script pour entraîner un classifier décidant du codage automatique avec le top-1")
    parser.add_argument("model_dir", nargs='?', type=str, default=None,
                        help="dossier du training")
    args = parser.parse_args()
    
    if hasattr(args, 'model_dir') and args.model_dir:
        save_dir = os.path.abspath(args.model_dir)
    else:
        save_dir = get_last_local_training_dir(TRAININGS_LOCAL_DIR)

    # add log in training dir
    test_log_file = os.path.join(save_dir, 'topk.log')
    formatter = logging.Formatter(log_config['formatters']['simple']['format'])
    ch = logging.FileHandler(test_log_file)
    ch.setLevel('DEBUG')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger_name = 'Top1AutoClassifier'
    logger = logging.getLogger(logger_name)
    
    config = load_config(save_dir)
    sync_with_minio = config['minio']['sync']
    metric_to_maximize = config['top1_classifier']['metric_to_maximize']

    # re-load all elements
    logger.info('Loading data')
    data_train = pd.read_csv(os.path.join(save_dir, 'optim_train_results.csv'),
                             converters={
                                 'top_k_codes': lambda x: x[1:-1].replace("'", "").replace(' ', '').split(','),
                                 'top_k_similarities': lambda s: [float(x.strip(' []')) for x in s.split(',')]})
    data_test = pd.read_csv(os.path.join(save_dir, 'optim_test_results.csv'),
                            converters={
                                 'top_k_codes': lambda x: x[1:-1].replace("'", "").replace(' ', '').split(','),
                                 'top_k_similarities': lambda s: [float(x.strip(' []')) for x in s.split(',')]})

    data_train, data_validation = train_test_split(data_train, test_size=0.4)
    for data in [data_train, data_validation, data_test]:
        # stats on diff between 1 and 2 when gt is 1
        data['top_1_ok'] = data['gt'] == data['top_k_codes'].map(lambda x: x[0])
        data['top_1_score'] = data['top_k_similarities'].map(lambda x: x[0])
        data['top_2_score'] = data['top_k_similarities'].map(lambda x: x[1])
        data['top_12_diff'] = data['top_k_similarities'].map(lambda x: x[0] - x[1])
        data['top_12_normdiff'] = data['top_k_similarities'].map(lambda x: (x[0] - x[1])/x[0])
        data['top_12_div'] = data['top_k_similarities'].map(lambda x: x[0] / x[1])
        data.dropna(inplace=True)

    scaler = StandardScaler().fit(pd.concat([
        data_train[['top_1_score', 'top_2_score', 'top_12_diff', 'top_12_normdiff', 'top_12_div']],
        data_validation[['top_1_score', 'top_2_score', 'top_12_diff', 'top_12_normdiff', 'top_12_div']],
        data_test[['top_1_score', 'top_2_score', 'top_12_diff', 'top_12_normdiff', 'top_12_div']]
    ]))
        
    # test models
    models = [
        # SVC(kernel="linear", C=0.025, probability=True),    -> long to train and not very good
        # SVC(gamma=2, C=1, probability=True),                -> long to train and not very good
        # GaussianProcessClassifier(1.0 * RBF(1.0)),          -> killed for memory
        DecisionTreeClassifier(max_depth=5),
        RandomForestClassifier(max_depth=5),
        MLPClassifier(alpha=1, max_iter=1000),
        AdaBoostClassifier(),
        GaussianNB()
    ]
    data_train['top_1_ok'].iloc[0] = True
    results = []
    for clf in models:
        logger.info(f'Optimizing model {clf.__class__.__name__}')
        X_train = data_train[['top_1_score', 'top_2_score', 'top_12_diff', 'top_12_normdiff', 'top_12_div']].values
        X_train = scaler.transform(X_train)
        y_train = data_train['top_1_ok'].values
        clf.fit(X_train, y_train)
        score_train = clf.score(X_train, y_train)

        X_validation = data_validation[['top_1_score', 'top_2_score', 'top_12_diff', 'top_12_normdiff', 'top_12_div']].values
        X_validation = scaler.transform(X_validation)
        y_validation = data_validation['top_1_ok'].values
        score_validation = clf.score(X_validation, y_validation)
        thresholds = np.arange(0.5,1,0.025)
        precisions = []
        recalls = []
        f1s = []
        for threshold in thresholds:
            precision, recall, f1 = calculate_metrics(clf, X_validation, y_validation, threshold)
            precisions.append(precision)
            recalls.append(recall),
            f1s.append(f1)

        best_ind, best_score = sorted(list(enumerate(eval(metric_to_maximize + 's'))), 
                                      key=lambda x: x[1], reverse=True)[0]

        result = {
            'clf': clf.__class__.__name__,
            'score_train': score_train,
            'score_validation': score_validation,
            'thresholds': {
                'thresholds': thresholds.tolist(),
                'precisions': precisions,
                'recalls': recalls,
                'f1s': f1s
            },
            'best_threshold': list(thresholds)[best_ind],
            'best_precision': precisions[best_ind],
            'best_recall': recalls[best_ind],
            'best_f1': f1s[best_ind]
        }
        logger.info(result)
        results.append(result)
        
    # choose best classifier
    best_classif, best_result = sorted(list(zip(models, results)), 
                                       key=lambda x: x[1]['best_'+metric_to_maximize], 
                                       reverse=True)[0]
    logger.info(f'Calculating final perfs with model {best_classif.__class__.__name__}')
    X_test = data_test[['top_1_score', 'top_2_score', 'top_12_diff', 'top_12_normdiff', 'top_12_div']].values
    X_test = scaler.transform(X_test)
    y_test = data_test['top_1_ok'].values
    score_test = best_classif.score(X_test, y_test)
    test_precision, test_recall, test_f1 = calculate_metrics(best_classif, 
                                                             X_test, y_test, 
                                                             best_result['best_threshold'])
    
    logger.info('Saving models and results')
    with open(os.path.join(save_dir, 'top1_scaler.pkl'), 'wb') as f:
        pickle.dump(scaler, f)
    with open(os.path.join(save_dir, 'top1_classifier.pkl'), 'wb') as f:
        pickle.dump(best_classif, f)
    final_result = {
        'maximized_metric': metric_to_maximize,
        'models': results,
        'best_model' : best_classif.__class__.__name__,
        'threshold' : best_result['best_threshold'],
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'scaler_file': os.path.join(save_dir, 'top1_scaler.pkl'),
        'model_file': os.path.join(save_dir, 'top1_classifier.pkl')
    }
    with open(os.path.join(save_dir, "top1_training.json"), "w") as f:
        json.dump(final_result, f)
    logger.info(final_result)
    
    config['top1_classifier']['scaler_file'] = 'top1_scaler.pkl'
    config['top1_classifier']['model_file'] = 'top1_classifier.pkl'
    config['top1_classifier']['threshold'] = float(best_result['best_threshold'])
    save_config(config, save_dir)
        
    # push on minio
    if sync_with_minio:
        push_to_minio(save_dir)
    
    # test validator class
    logger.info('Testing validator')
    validator = Top1ValidatorModel(save_dir)
    validated = validator.validate_top1(data_test['top_k_similarities'].values)
    tp = (validated == 1) & (y_test == 1)
    tn = (validated == 0) & (y_test == 0)
    fp = (validated == 1) & (y_test == 0)
    fn = (validated == 0) & (y_test == 1)
    precision = sum(tp) / max(1, (sum(tp) + sum(fp)))
    recall = sum(tp) / max(1, (sum(tp) + sum(fn)))
    f1 = 2 * (precision*recall)/max(1, (precision+recall))
    if f1 != test_f1:
        logger.error('The loaded model does not show the same performances as the saved model. There may have been a problem')
    else:
        logger.info('Validator is ok')