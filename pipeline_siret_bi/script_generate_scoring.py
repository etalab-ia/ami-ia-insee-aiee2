#!/usr/bin/env python3
"""
2021/01/19

Génère des métriques et des graphique sur les
performances du codage automatique et les topk

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
import seaborn as sns
import s3fs

import data_import.bdd as bdd
import elastic as elastic

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import precision_recall_curve

from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from scipy import interp
from sklearn.metrics import roc_auc_score

import scikitplot as skplt

from script_metamodel_optimisation import *


def reformat_topk_df_to_topk_dict(df):
    """
    Wrapper qui met dans le bon format les topk pour calculer la précision
    
    """
    topk = {}
    for idx, row in  df.iterrows():
        topk[str(idx)] = [ {"siret":row['top_1_siret']}
                            ,{"siret":row['top_2_siret']}
                            ,{"siret":row['top_3_siret']}
                            ,{"siret":row['top_4_siret']}
                            ,{"siret":row['top_5_siret']}
                            ,{"siret":row['top_6_siret']}
                            ,{"siret":row['top_7_siret']}
                            ,{"siret":row['top_8_siret']}
                            ,{"siret":row['top_9_siret']}
                            ,{"siret":row['top_10_siret']}
                            ]
    return topk


def plot_confusion_matrix(cf_matrix):
    """
    Plot de matrice de confusion lisible
    """
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cf_matrix.flatten()/np.sum(cf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names, group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2,2)
    figure = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')
    return figure


def main():
    """
    On génère des graphiques sur les perfs du méta modèle et des topk
    """
    
    # Chargement donnée
    with open(os.path.join(os.path.dirname(__file__), 'logging.conf.yaml'), 'r') as stream:
        log_config = yaml.load(stream, Loader=yaml.FullLoader)
    logging.config.dictConfig(log_config)
    logger = logging.getLogger(__file__)
    
    logger.info(" begin ")

    #######################
    #   Variables du script
    #######################
    work_dir = 'trainings/2021-02-18_25'
    gt_pickle_file = os.path.join(work_dir, 'df_solution_test.p')
    
    minio_endpoint = 'http://minio.ouest.innovation.insee.eu'
    minio_path_to_bdd_settings_file = 's3://ssplab/aiee2/data/settings.yml'
    rp_table = "rp_final_2019"
    sirus_table = "siret_2019"
    precalculated_topk_table = 'bi_topk_test_refacto_final'

    #################
    #   Création du fichier de données pour analyse
    #################

    # load gt data
    with open(gt_pickle_file, "rb") as input_file:
        df_ground_truth = pickle.load(input_file)
    df_ground_truth = df_ground_truth.set_index("cabbi")
    
    # get data from BDD
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': minio_endpoint})
    driver_bdd = bdd.PostGre_SQL_DB(fs=fs, 
                                    fs_path_to_settings=minio_path_to_bdd_settings_file)
    
    logger.info("Chargement Prédictions Top-k...")
    sql: str = f"SELECT * FROM {precalculated_topk_table} LIMIT 10000;"
    df = driver_bdd.read_from_sql(sql)
    df = df.set_index("cabbi")
    
    # Récupération des target / ground truth auto codage
    for idx, row in df.iterrows():
        solution_sirus = df_ground_truth.loc[idx]['sirus_id']
        solution_nic = df_ground_truth.loc[idx]['nic']
        solution_id_sirus = solution_sirus + solution_nic
        if row["top_1_siret"] == solution_id_sirus:
            df.loc[idx, "target"] = 1
        else:
            df.loc[idx, "target"] = 0
            
            
    # Selection observation pour voir manuellement des erreur
    topk_metrics_dict = metric_top(reformat_topk_df_to_topk_dict(df), df_ground_truth)
    list_3 = []
    list_5 = []
    list_10 = []
    na= []
    for key in topk_metrics_dict.keys():
        if topk_metrics_dict[key] == "not_found":
            na.append(key)
            continue
        if int(topk_metrics_dict[key]) <= 3 and int(topk_metrics_dict[key]) > 1:
            list_3.append(key)
        if int(topk_metrics_dict[key]) <= 5 and int(topk_metrics_dict[key]) > 3:
            list_5.append(key)
        if int(topk_metrics_dict[key]) <= 10 and int(topk_metrics_dict[key]) > 5:
            list_10.append(key)
    
    # recuperation des BI
    sql: str = f"SELECT * FROM {rp_table} WHERE cabbi IN {tuple(list_3)};"
    bi = driver_bdd.read_from_sql(sql)
    bi = bi.set_index("cabbi")
            
    dd = {}
    for key in list_3:
        siret_to_fetch = []
        df_sir = pd.DataFrame()
        for n in range(9):
            # recuperation des entreprises
            siret_to_fetch = df.loc[key, f"top_{n+1}_siret"]
            sql: str = f"SELECT * FROM {sirus_table} WHERE siret = '{siret_to_fetch}';"
            sir = driver_bdd.read_from_sql(sql)
            df_sir = df_sir.append(sir,ignore_index=True)
        dd[key] = {}
        dd[key]["bi"] = bi.loc[key]
        dd[key]["sirets"] = df_sir
        dd[key]["target"] = topk_metrics_dict[key]
        
    # On dump pour pouvoir regarder le df dans un jupyter    
    with open(os.path.join(work_dir, 'dataset_erreur.p'), 'wb') as f:
        pickle.dump(dd, f)
    
    ###############
    #   Create plots
    ###############

    # On regarde les seuils sur une partie du jeu de donnée puis
    # on test sur une autre partie du jeu de donnée
    X_train, X_test, y_train, y_test = train_test_split(df["top_1_codage_auto_proba"].values, df["target"].values, test_size=0.5, random_state = 0)
    
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(1):
        fpr[i], tpr[i], _ = roc_curve(y_train, X_train)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_train, X_train)
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    plt.figure()
    lw = 2
    plt.plot(fpr[0], tpr[0], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()
    plt.savefig(os.path.join(work_dir, "roc.png"))
    
    plt.figure()
    skplt.metrics.plot_roc(y_train, np.swapaxes(np.vstack([1-X_train, X_train]), 0, 1))
    plt.savefig(os.path.join(work_dir, "plot_roc.png"))
    
    plt.figure()
    skplt.metrics.plot_precision_recall(y_train, np.swapaxes(np.vstack([1-X_train, X_train]), 0, 1))
    plt.savefig(os.path.join(work_dir, "plot_precision_recall.png"))
    
    plt.figure()
    skplt.metrics.plot_calibration_curve(y_train, [X_train], ["Xgboost"])
    plt.savefig(os.path.join(work_dir, "plot_calibration_curve.png"))
    
    # Précision/recall plot
    precision, recall, thresholds = precision_recall_curve(y_train, X_train)
    dff = pd.DataFrame(list(zip(precision, recall, thresholds)), columns=["precision","recall","seuil"])
    dff = dff.sort_values(by=['seuil'])
    fig, ax = plt.subplots()
    fig.set_size_inches(11.7, 8.27)
    sns.lineplot(data=dff, x = 'seuil', y='precision', color='b',legend='brief', label=str("precision"))
    ax.set(ylim=(0, 1))
    sns.lineplot(data=dff, x = 'seuil', y='recall', color = 'orange', legend='brief', label=str("recall"))
    ax.legend(bbox_to_anchor=(0.175, 0.125))
    plt.ylabel('values')
    plt.savefig(os.path.join(work_dir, "pr_curve_2.png"))
    
    
    # Matrice de confusion pour 80% de précision
    for precision in [0.8, 0.9, 0.95]:
        logger.info(f"precision {precision}_geo")
        threshold = automatic_threshold(y_train, X_train, precision)
        plt.figure()
        y_round = (X_test >= threshold).astype('int')
        values, counts = np.unique(y_round, return_counts=True)
        tn, fp, fn, tp = confusion_matrix(y_test, y_round).ravel()
        df_matrix = confusion_matrix(y_test, y_round)
        fig = plot_confusion_matrix(df_matrix)
        fig.set_ylim([0,2])
        fig.figure.savefig(os.path.join(work_dir, f'mc_{int(precision*100)}_geo.png'))
        logger.info(f"seuil {threshold}")
    
    plt.figure()
    df_matrix = confusion_matrix(df["target"].values, df["top_1_codage_auto"].values)
    fig = plot_confusion_matrix(df_matrix)
    fig.set_ylim([0,2])
    fig.figure.savefig(os.path.join(work_dir, 'confusion_mat.png'))
    
    
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    main()