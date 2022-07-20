#!/usr/bin/env python3
"""
2021/01/19

Script qui calcul alpha et beta.
Script d'entrainement du meta model pour coadge automatique

Auteur: Brivaël Sanchez
"""
import sys
sys.path.append("..")
import os
import os.path
from os import path
import pandas as pd
import numpy as np
import shutil
import glob
import yaml
import getopt
import pickle
from datetime import date
import random
import json
import logging
import logging.config
import xgboost
from xgboost import XGBClassifier
from numpy import load
import s3fs

import data_import.bdd as bdd
import elastic as elastic

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import precision_recall_curve

from dataclasses import dataclass


####################
#   Calcul de métriques
####################
def metric_top(topk, df_groundtruth) -> int:
    """
    Génere les métriques topk.
    
    :params topk: dict pour clé cabbi et pour value list de siret trié par similarité cosinus
    :params df_groundtruth: df des cabbi avec les target
    
    :returns top_metric: dict ayant pour clé une position et pour valeur le nb de siret target
    """
    logger = logging.getLogger(os.path.basename(__file__))
    logger.info("compute_metrics")
    
    count = len(topk.keys())
    
    top_metric = {}
    dict_cabbi_top_is = {}
    for i in range(1, 200):
        top_metric[str(i)] = 0
    top_metric["not_found"] = 0
        
    for key in topk.keys():
        if key in df_groundtruth.index:
            solution_sirus = df_groundtruth.loc[key]['sirus_id']
            solution_nic = df_groundtruth.loc[key]['nic']
            solution_id_sirus = solution_sirus + solution_nic

            for idx, item in enumerate(topk[key]):
                is_found = False
                if item['siret'] == solution_id_sirus:
                    top_metric[str(idx + 1)] += 1
                    dict_cabbi_top_is[key] = str(idx + 1)
                    is_found = True
                    break
            if not is_found:
                top_metric["not_found"] += 1
                dict_cabbi_top_is[key] = "not_found"
        else:
            count -= 1
            logger.info(f"Le BI {key} n'est pas présent dans df_groundtruth")
            continue
                    
    value_top_1  = top_metric["1"]    
    value_top_3 = 0
    value_top_5 = 0
    value_top_10 = 0
    value_top_20 = 0
    value_top_50 = 0
    
    for i in range (1,3 + 1):
        value_top_3  += top_metric[str(i)] 
        
    for i in range (1,5 + 1):
        value_top_5  += top_metric[str(i)] 
        
    for i in range (1,10 + 1):
        value_top_10  += top_metric[str(i)] 

        
    logger.info(f"count is  {count}")
    logger.info(f"top1 is {value_top_1} and so {(value_top_1 / count) * 100}%")
    logger.info(f"top3 is {value_top_3} and so {(value_top_3 / count) * 100}%")
    logger.info(f"top5 is {value_top_5} and so {(value_top_5 / count) * 100}%")
    logger.info(f"top10 is {value_top_10} and so {(value_top_10 / count) * 100}%")

    
    return dict_cabbi_top_is 

   
####################
#   optimisation
####################
def topk_maker(dict_candidat, coeff_nb_fois_trouve, coeff_naf_score):
    """
    Réordonne le topk selon la similarité
    
    :params dict_candidat: dict du topk
    :params coeff_nb_fois_trouve: hyperparametre 
    :params coeff_naf_score: hyperparametre
    
    :return:  dict candidat ordonné par similarité
    """
    logger = logging.getLogger(os.path.basename(__file__))
    dict_topk = {}
    logger.info(coeff_nb_fois_trouve)
    logger.info(coeff_naf_score)
    for key in dict_candidat.keys():
        for i, candidat in enumerate(dict_candidat[key]):
            dict_candidat[key][i]["similarite_final"] = dict_candidat[key][i]["similarite"] + coeff_nb_fois_trouve * dict_candidat[key][i]["nb_fois_trouve"] + coeff_naf_score * dict_candidat[key][i]["naf_score"]
        dict_topk[key] = sorted(dict_candidat[key], key=lambda k: k['similarite_final'], reverse=True)
        
    return dict_topk


def objective(trial, dict_candidat, df_solution):
    """
    Fonction objectif à minimiser

    :param trial: objet trial
    :param dict_candidat: données input (top-k pré meta-modèle)
    :param df_solution: ground truth
    """
    coeff_nb_fois_trouve = trial.suggest_uniform('coeff_nb_fois_trouve', 0, 1)
    coeff_naf_score = trial.suggest_uniform('coeff_naf_score', 0, 1)
    
    dict_sim_final = topk_maker(dict_candidat, coeff_nb_fois_trouve, coeff_naf_score)
    dict_score = metric_top(dict_sim_final, df_solution)
    count = len(dict_score.keys())
    dict_top = metric_top_to_result_dict(dict_score)
    score = (dict_top["1"] / count) * 100
    
    return 1 - score

###################
#   Fonctions de conversion
###################
def metric_top_to_result_dict(metric_top_score_dict, force_keys = None):
    """
    Produit un dict avec le nb d'occurence pour chaque top 
    
    :params metric_top_score_dict: sortie de metric_top
    :params force_keys: clés à garder (les k d'intéret)
    
    :returns: dict
    """
    dict_top = {}
    for i in range (1,11):
        dict_top[str(i)] = 0
    dict_top["not_found"] = 0
    
    if force_keys is None:
        keys = metric_top_score_dict.keys()
    else:
        keys = force_keys
    
    for key in keys:
        dict_top[str(metric_top_score_dict[key])] += 1
    return dict_top
    

def make_podium_score(dict_t):
    """
    Créer le score cumulatif pour les topk
    
    :params dict_t: dict des occurence par topk (sortie de metric_top_to_result_dict)
    
    :return:  dict_podium
    """
    dict_podium = {}
    dict_podium["value_top_1"] = dict_t["1"]
    dict_podium["value_top_3"] = 0
    dict_podium["value_top_5"] = 0
    dict_podium["value_top_10"] = 0
        
    for i in range (1,3 + 1):
        dict_podium["value_top_3"]  += dict_t[str(i)] 
        
    for i in range (1,5 + 1):
        dict_podium["value_top_5"]  += dict_t[str(i)] 
        
    for i in range (1,10 + 1):
        dict_podium["value_top_10"]  += dict_t[str(i)]
        
    return dict_podium


def prepare_topk_df_to_trial_format(df, df_bi_naf, df_siret_naf):
    """
    Conversion des données stockées dans la db via script_runtime vers les trials

    :param df: input df
    :param df_bi_naf: df contenant les prédictions naf associées aux données
    :param df_siret_naf: df contenant les données SIRET
    :return dict
    """
    dict_top = df.to_dict('index')
    dict_top_trial = {}
    for key in dict_top.keys():
        dict_top_trial[key] = []
        if key in df_bi_naf.index:
            list_naf_candidat = list(df_bi_naf.loc[key].index)
        else:
            list_naf_candidat = []
        for i in range(1,10 + 1):
            candidat = {"siret" : dict_top[key][f"top_{i}_siret"]
                        , "similarite" : dict_top[key][f'top_{i}_similarite']
                        , "nb_fois_trouve" : dict_top[key][f'top_{i}_nb_fois_trouve']}

            naf_candidat = df_siret_naf.loc[candidat["siret"]]["ape"]

            if naf_candidat in list_naf_candidat:
                candidat["naf_score"] = df_bi_naf.loc[key].loc[naf_candidat]["naf_score"]
            else:
                candidat["naf_score"] = 0

            dict_top_trial[key].append(candidat)

    return dict_top_trial


def topk_res_to_meta_model_format(dict_topk):
    """
    Retourne un dataframe dans le bon format pour l'entrainement du meta model.
    Supression de toutes les entrées sauf pour le top1 et le top2
    
    :params dict_topk: dict topk
    
    :return: dataframe avec le top1 et le top2 en colonne
    """
    dict_best_only_first = {}
    for key in dict_topk.keys():
        dict_best_only_first[key] = dict_topk[key][0]
    
    dict_best_only_second = {}
    for key in dict_topk.keys():
        dict_best_only_second[key] = dict_topk[key][1]
    df_topk_only_first = pd.DataFrame.from_dict(dict_best_only_first, orient='index')
    df_topk_only_second = pd.DataFrame.from_dict(dict_best_only_second, orient='index')
    df_topk_first_second = df_topk_only_first.merge(df_topk_only_second, how='left', left_index=True, right_index=True, validate= 'one_to_one', suffixes=("_1", "_2"))
    df_topk_first_second["target"] = 0
    df_topk_first_second.sort_index(axis=1, inplace=True)
    return df_topk_first_second


#################
#   Calcul auto du threshold
#################
@dataclass(frozen=True)
class Threshold:
    value: float
    precision: float 
    recall: float
        
def automatic_threshold(y_true, y_scores, min_precision):
    """
    
    Recherche le seuil de décision ayant une précison minimal donnée par 
    "min_precision" avec le meilleur recall

    :params y_true: list target
    :params y_scores: list de prediction sous forme de proba
    :params min_precision: précision recherché
    
    :return: un seuil de décision (entre 0 et 1)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
    list_threshold = []
    for a,b,c in zip(precision, recall, thresholds):
        list_threshold.append(Threshold(recall = b, precision = a, value = c))
    filter_list = list(filter(lambda x: x.precision >= min_precision, list_threshold))
    # la précision la plus haute est en dessous de la précision recherché, on renvoie un seuil par défaut
    if len(filter_list) > 0:
        threshold_point = max(filter_list, key=lambda x: x.recall)
    else:
        return 0.5
    return threshold_point.value


##################
#   Runtime codage auto
##################
def codage_automatique(topk, meta_model, threshold):
    """
    Codage automatique :
    Production via le méta model d'une probabilité que le top1 du topk soit le bon siret :
    Si la probabilité est supérieur au seuil, on code.
    
    :params topk: dictionnaire contenant les topf
    :params meta_model: meta modele pour le codage automatique, généré par le script meta_model_otpimisation.py
    :params threshold: seuil de décision entre (0 et 1)
    
    :return: topk avec un nouveau champs codage automatique
    """
    df = topk_res_to_meta_model_format(topk)
    df.drop(["target","siret_1","siret_2","cabbi_1","cabbi_2"], axis=1, inplace=True)
    df["ecart"] =  df.apply(lambda x: x["similarite_final_1"] - x["similarite_final_2"],axis=1) 
    cols_to_keep = ['naf_score_1', 'naf_score_2', 'nb_fois_trouve_1', 'nb_fois_trouve_2', 'similarite_1', 'similarite_2', 'similarite_final_1', 'similarite_final_2', 'ecart']
    df = df[cols_to_keep]
    pred = meta_model.predict_proba(df)
    code = (pred[:,1] >= threshold).astype('int')
    df["code"] = code
    df["code_prob"] = pred[:,1]

    for idx, row in df.iterrows():
        topk[idx][0]['codage_auto'] = df.loc[idx, "code"]
        topk[idx][0]['codage_auto_proba'] = df.loc[idx, "code_prob"]
        
    return topk



def main():
    with open(os.path.join(os.path.dirname(__file__), 'logging.conf.yaml'), 'r') as stream:
        log_config = yaml.load(stream, Loader=yaml.FullLoader)
    logging.config.dictConfig(log_config)
    logger = logging.getLogger(os.path.basename(__file__))
    
    logger.info(" begin ")

    #######################
    #   Variables du script
    #######################
    work_dir = 'trainings/2021-02-18_25'
    gt_pickle_file = os.path.join(work_dir, 'df_solution_eval.p')
    
    minio_endpoint = 'http://minio.ouest.innovation.insee.eu'
    minio_path_to_bdd_settings_file = 's3://ssplab/aiee2/data/settings.yml'
    sirus_table = "siret_2019"
    naf_proj_table = "naf_projections_2019"
    precalculated_topk_table = 'bi_topk_test_refacto'

    n_trials = 2
    n_hours = 1

    target_precision = 0.8
    
    #################
    #   Run
    #################
    # load gt data
    with open(gt_pickle_file, "rb") as input_file:
        df_ground_truth = pickle.load(input_file)
    df_ground_truth = df_ground_truth.set_index("cabbi")

    # get data from BDD
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': minio_endpoint})
    driver_bdd = bdd.PostGre_SQL_DB(fs=fs, 
                                    fs_path_to_settings=minio_path_to_bdd_settings_file)
    
    
    logger.info("Chargement projections NAF...")
    col_naf = "cabbi, "
    col_naf += ", ".join([f"naf_code_{i}" for i in range(10)]) + ", "
    col_naf += ", ".join([f"naf_score_{i}" for i in range(0,10)])
    sql: str = f"SELECT {col_naf} FROM {naf_proj_table}"
    df_bi_naf = driver_bdd.read_from_sql(sql)
    logger.info(f"wide_to_long {naf_proj_table}...")
    df_bi_naf = pd.wide_to_long(df_bi_naf, stubnames=["naf_code",'naf_score'], i='cabbi', j = "dummy",sep="_" ).reset_index().drop("dummy",axis=1).set_index(["cabbi","naf_code"])
    
    logger.info("Chargement codes APE...")
    sql: str = f"SELECT siret, ape FROM {sirus_table}"
    df_siret_naf = driver_bdd.read_from_sql(sql).set_index("siret")
    # logger.info(df_siret_naf.head())
    
    logger.info("Chargement Prédictions Top-k...")
    sql: str = f"SELECT * FROM {precalculated_topk_table} LIMIT 100;"
    df = driver_bdd.read_from_sql(sql)
    df = df.set_index("cabbi")
    
    ################
    #   Optimisation
    ################
    logger.info("OPTIMISATION COEFFS ALPHA / BETA")
    train_data, test_data = train_test_split(df, test_size=0.4)

    train_data = prepare_topk_df_to_trial_format(train_data, df_bi_naf, df_siret_naf)
    test_data = prepare_topk_df_to_trial_format(test_data, df_bi_naf, df_siret_naf)
    
    study = optuna.create_study(study_name="test", direction='minimize')
    study.optimize(lambda trial : objective(trial, train_data, df_ground_truth),
                   n_trials=n_trials,
                   timeout=n_hours*60*60)
    
    best_trial = study.best_trial
    
    ################
    #   Exploitation résultats
    ################
    logger.info("RESULTAT OPTIMISATION")
     
    # meilleur résultat
    logger.info("_____meilleur coef_____")
    coeff_nb_fois_trouve_best = best_trial.params["coeff_nb_fois_trouve"]
    coeff_naf_score_best = best_trial.params["coeff_naf_score"]
    best_predictions = topk_maker(test_data, coeff_nb_fois_trouve_best, coeff_naf_score_best)
    best_score = metric_top(best_predictions, df_ground_truth)
    best_res_dict = metric_top_to_result_dict(best_score)
    best_podium = make_podium_score(best_res_dict)

    # sans les ajouts
    logger.info("____coef témoin____")
    vanilla_predictions = topk_maker(test_data, 0, 0)
    vanilla_score = metric_top(vanilla_predictions, df_ground_truth)
    vanilla_res = metric_top_to_result_dict(vanilla_score, force_keys=best_score.keys())
    vanilla_podium = make_podium_score(vanilla_res)
                   
    logger.info("_____Comparatif_____")     
    count = len(best_score.keys())
                   
    logger.info(f"count is {count}")
    logger.info(f'top1 is {vanilla_podium["value_top_1"]} and so {(vanilla_podium["value_top_1"] / count) * 100}%')
    logger.info(f'top1 is {best_podium["value_top_1"]} and so {(best_podium["value_top_1"] / count) * 100}%')
    
    logger.info("---------------")           
    logger.info(f'top3 is {vanilla_podium["value_top_3"]} and so {(vanilla_podium["value_top_3"] / count) * 100}%')
    logger.info(f'top3 is {best_podium["value_top_3"]} and so {(best_podium["value_top_3"] / count) * 100}%')
    
    logger.info("---------------")
    logger.info(f'top5 is {vanilla_podium["value_top_5"]} and so {(vanilla_podium["value_top_5"] / count) * 100}%')
    logger.info(f'top5 is {best_podium["value_top_5"]} and so {(best_podium["value_top_5"] / count) * 100}%')
    
    logger.info("---------------")
    logger.info(f'top10 is {vanilla_podium["value_top_10"]} and so {(vanilla_podium["value_top_10"] / count) * 100}%')
    logger.info(f'top10 is {best_podium["value_top_10"]} and so {(best_podium["value_top_10"] / count) * 100}%')

    with open(os.path.join(work_dir, f'best_param_optuna.json'), 'w') as f:
        json.dump(best_trial.params, f)
        
    
    #####################
    #   Modele codage automatique
    #####################
    logger.info("TRAINING MODELE CODAGE AUTO")
    # format input for model
    df_topk_first_second = topk_res_to_meta_model_format(best_predictions)
    # add ground truth
    for idx, row in df_topk_first_second.iterrows():
        solution_sirus = df_ground_truth.loc[idx]['sirus_id']
        solution_nic = df_ground_truth.loc[idx]['nic']
        solution_id_sirus = solution_sirus + solution_nic
        if row["siret_1"] == solution_id_sirus:
            df_topk_first_second.loc[idx, "target"] = 1
    
    target = df_topk_first_second["target"].values
    df_topk_first_second.drop(["target","siret_1","siret_2"], axis=1, inplace=True)
    df_topk_first_second["ecart"] =  df_topk_first_second.apply(lambda x: x["similarite_final_1"] - x["similarite_final_2"],axis=1)
    
            
    # logger.info(df_topk_first_second.head())
    logger.info(f"nb targets : {np.sum(target)}/{len(df_topk_first_second)}")
    # logger.info(df_topk_first_second.columns)
    
    # create datasets
    X_train, X_test, y_train, y_test = train_test_split(df_topk_first_second, target, 
                                                        test_size=0.3)
    X_train, X_eval, y_train, y_eval = train_test_split(X_train, y_train, 
                                                        test_size=0.4)
    # train classifier
    clf = XGBClassifier().fit(X_train, y_train)
    
    logger.info("OPTIMIZING THRESHOLD")

    p_pred_eval = clf.predict_proba(X_eval)[:,1]
    
    # Métrique
    precision, recall, thresholds = precision_recall_curve(y_eval, p_pred_eval)
    dff = pd.DataFrame(list(zip(precision, recall, thresholds)), columns=["precision","recall","seuil"])
    dff = dff.sort_values(by=['seuil'])
    fig, ax = plt.subplots()
    fig.set_size_inches(11.7, 8.27)
    sns.lineplot(data=dff, x = 'seuil', y='precision', color='b',legend='brief', label=str("precision"))
    ax.set(ylim=(0, 1))
    sns.lineplot(data=dff, x = 'seuil', y='recall', color = 'orange', legend='brief', label=str("recall"))
    ax.legend(bbox_to_anchor=(0.175, 0.125))
    plt.ylabel('values')
    plt.savefig(os.path.join(work_dir, "pr_curve.png"))

    # optimiser threshold
    precisions = [0.8, 0.9, 0.95]
    if target_precision not in precisions:
        precisions.append(target_precision)

    thresholds = []
    for precision in precisions:
        thresholds.append(automatic_threshold(y_eval, p_pred_eval, precision))
        logger.info(f"precision {precision}: {thresholds[-1]}")
    
    # calcul performances finales
    p_pred_test = clf.predict_proba(X_test)
    for threshold in thresholds:
        y_pred_test = (p_pred_test[:,1] >= threshold).astype('int')
        report = classification_report(y_test, y_pred_test)
        conf_mat = confusion_matrix(y_test, y_pred_test)
        if conf_mat.shape[0] == 2:
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_test).ravel()
        else:
            if y_test[0] == 0:
                tn, fp, fn, tp = len(y_test), 0, 0, 0
            else:
                tn, fp, fn, tp = 0, 0, 0, len(y_test)
        logger.info(report)
        logger.info(f"TN:{tn}  FP:{fp}  FN:{fn}  TP:{tp} ")

    # save results
    target_threshold = thresholds[precisions.index(target_precision)]
    with open(os.path.join(work_dir, f'threshold.p'), 'wb') as f:
        pickle.dump(target_threshold, f)
        
    with open(os.path.join(work_dir, f'meta_model.p'), 'wb') as f:
        pickle.dump(clf, f)
        

if __name__ == '__main__':
    import optuna
    from optuna import Trial
    import matplotlib.pyplot as plt
    import seaborn as sns

    main()