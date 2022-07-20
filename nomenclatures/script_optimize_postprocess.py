#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# optimize_postprocess.py training_dir weight_dir
#     training_dir : dossier du training. 
#                    Défault : training le plus récent dans config['local']['trainings_dir']/config['data']['nomenclature']['name']
#     weights_dir : nom du dossier de poids à utiliser dans le training/train_weights
#                   Défaut : le meilleur modèle dans training_dir
#
# script d'optimisation du postprocess, ie des poids associé aux 2 postprocess disponibles :
#    - ajout des similarité des noeuds parents
#    - boost de la similarité par une mesure de similarité textuelle basée sur le compte de trigrammes
# la valeur optimisée est le top-k avec k=config['post_process']['top_X_to_optimize']
#
# le script utilise optuna:  pour générer des valeurs pour les 2 paramêtres, calculer les performances sur un 
# - on charge le modèle
# - on charge les données de test du training (jamais vu par le modèle)
# - on sépare en train/test à nouveau (50/50)
# - optuna génère des valeurs pour les 2 paramêtres, calcule les performances sur le train et sauvegarde le tout
# - on s'arrète après config['post_process']['optim_timeout_in_min'] minutes ou config['post_process']['nb_trials'] essais
# - on update le config.yaml du training avec les meilleures valeurs trouvées
# - on calcule les performances sur le test avec les meilleurs valeurs de parametres, et on sauvegarde le tout en local et en remote
#
# utilise config.yaml dans le dossier de training
#
# @author cyril.poulet@starclay.fr
# @date: oct 2020

import sys
sys.path.append('..')
import os
import yaml
import json
import shutil
import logging.config
import s3fs
from tensorflow import keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import optuna
from optuna import Trial

from data_import.bdd import PostGre_SQL_DB
from training_utils import load_config, save_config, push_to_minio, get_trainings_dir, get_last_local_training_dir, get_best_savedmodel
from script_run_top_k import load_model_from_save_dir, run_top_k_on_test


if __name__ == "__main__":

    """
    Script permettant d'optimiser les paramêtres de post-process via une exploration de l'espace'

    """
    import argparse
    args = None
    
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
        description="Script pour optimiser le post-process")
    parser.add_argument("model_dir", nargs='?', type=str, default=None,
                        help="dossier du training")
    parser.add_argument("weights_dir", nargs='?', type=str, default=None,
                        help="nom du dossier de poids à utiliser dans le training/train_weights")
    args = parser.parse_args()
    
    if hasattr(args, 'model_dir') and args.model_dir:
        save_dir = os.path.abspath(args.model_dir)
    else:
        save_dir = get_last_local_training_dir(TRAININGS_LOCAL_DIR)
    if hasattr(args, 'weights_dir') and args.weights_dir:
        best_weights = os.path.abspath(args.weights_dir)
    else:
        best_weights = get_best_savedmodel(save_dir)
        
        
    save_dir = get_last_local_training_dir(TRAININGS_LOCAL_DIR)
    best_weights = get_best_savedmodel(save_dir)

    # add log in training dir
    test_log_file = os.path.join(save_dir, 'topk.log')
    formatter = logging.Formatter(log_config['formatters']['simple']['format'])
    ch = logging.FileHandler(test_log_file)
    ch.setLevel('DEBUG')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger_name = 'NomenclatureTopKOptimizer'
    logger = logging.getLogger(logger_name)
    
    logger.info(f'Running post-process optimizer pour {save_dir}/{best_weights}')
    config = load_config(save_dir)
    sync_with_minio = config['minio']['sync']

    # re-load all elements
    logger.info('Loading all trained objects')
    try:
        nomenclature, data_cleaner, data_formatter, model, _ = load_model_from_save_dir(save_dir, best_weights)
        modified = False
        if nomenclature.projections is None:
            logger.info('building nomenclature projections')
            nomenclature.build_nomenclature_projections(
                lambda x : model.run_model_single_side(
                    [np.array([emb], dtype=np.float32) for emb in data_formatter.format_input(x)])[0])
            modified = True
        if nomenclature.ngram_hot_encodings is None:
            nomenclature.create_trigram_repr()
            modified = True
        if modified:
            data_formatter.save(os.path.join(save_dir, "batcher"))
            if sync_with_minio:
                logger.info('pushing to minio')
                push_to_minio(save_dir)
    except Exception as e:
        logger.error(f'Error loading objects : {e}')
        exit(-1)

    # loading test data
    logger.info('Loading list of test BIs')
    try:
        test_cabbis_df = pd.read_csv(os.path.join(save_dir, 'cabbi_test.csv'))
    except Exception as e:
        logger.error(f'Error loading list of test BIs : {e}')
        exit(-1)
        
    # split in train/test
    cabbi_optim, cabbi_test = train_test_split(test_cabbis_df, test_size=0.5)
    cabbi_optim.to_csv(os.path.join(save_dir, 'optim_train_cabbis.csv'))
    cabbi_test.to_csv(os.path.join(save_dir, 'optim_test_cabbis.csv'))
    
    def get_sql_request(config, cabbis_df):
        sql_request = config['data']['postgres_sql']
        cabbi_field = [t for t in sql_request.split(' ') if 'cabbi' in t][0].strip(',. ')
        list_of_cabbi = "('" + "','".join([str(v) for v in cabbis_df["cabbi"].values]) + "')"
        if 'where' in sql_request.lower():
            sql_request += ' AND '
        else:
            sql_request += ' WHERE '
        sql_request += f'{cabbi_field} IN {list_of_cabbi}'
        return sql_request
    
    # run optimizer
    optim_savedir = os.path.join(save_dir,'postprocess_optim')
    os.makedirs(optim_savedir, exist_ok=True)
    
    def objective(trial: Trial):
        """
        définit ce qui est fait pendant une passe:
        - génération de valeurs
        - calcul des top-k
        - sauvegarde du tout
        """
        if 'post_process' not in config:
            config['post_process'] = {}
        config['post_process']['alpha_tree_mod'] = trial.suggest_uniform("alpha_tree", 0, 1)
        config['post_process']['beta_str_sim_mod'] = trial.suggest_uniform("beta_str_sim", 0, 1)
        
        sql_request = get_sql_request(config, cabbi_optim)
        bdd = PostGre_SQL_DB()
        batch_size = config['trainings']['training_params']['batch_size']
        input_df = bdd.read_from_sql_with_chunksize(sql_request, batch_size)
        nb_docs = len(cabbi_optim["cabbi"].values)
        
        logger.info(f"Calculating top-k for alpha={config['post_process']['alpha_tree_mod']}, beta={config['post_process']['beta_str_sim_mod']}")
        global_results, _ = run_top_k_on_test(config, input_df, nb_docs,
                                              data_cleaner, data_formatter, model,
                                              to_csv_file=os.path.join(optim_savedir, f'test_results_{trial.number}.csv'))
        logger.info(f'Results: {global_results}')
        global_results['saved_model'] =  best_weights
        global_results['alpha_tree_mod'] = config['post_process']['alpha_tree_mod']
        global_results['beta_str_sim_mod'] = config['post_process']['beta_str_sim_mod']
        with open(os.path.join(optim_savedir, f'top_k_{trial.number}.json'), 'w') as f:
            json.dump(global_results, f)
            
        top_x_to_optimize = config['post_process']['top_X_to_optimize']
        trial.report(global_results['top_k_perc'][top_x_to_optimize], step=1)
        return global_results['top_k_perc'][top_x_to_optimize]
    
    #création du l'"étude" optuna
    study_name = 'postprocess_optim'
    maximum_time = config['post_process']['optim_timeout_in_min'] * 60   # seconds
    number_of_trials = config['post_process']['nb_trials']
    
    optuna.logging.enable_propagation()
    optuna.logging.disable_default_handler()
    study = optuna.create_study(study_name=study_name, direction='maximize')
    try:
        study.optimize(objective, n_trials=number_of_trials, timeout=maximum_time)
    except Exception as e:
        logger.error(e)
        exit(-1)
      
    # récupératio du résultat et modification de config
    df = study.trials_dataframe()
    df.to_json(os.path.join(optim_savedir, 'optimisation_results.json'))
    best_trial = study.best_trial
    if 'post_process' not in config:
        config['post_process'] = {}
    config['post_process']['alpha_tree_mod'] = best_trial.params["alpha_tree"]
    config['post_process']['beta_str_sim_mod'] = best_trial.params["beta_str_sim"]
    save_config(config, save_dir)
    shutil.copy(os.path.join(optim_savedir, f'test_results_{best_trial.number}.csv'), 
                os.path.join(save_dir, 'optim_train_results.csv'))
    
    
    # get final perfs
    sql_request = get_sql_request(config, cabbi_test)
    bdd = PostGre_SQL_DB()
    batch_size = config['trainings']['training_params']['batch_size']
    input_df = bdd.read_from_sql_with_chunksize(sql_request, batch_size)
    nb_docs = len(cabbi_test["cabbi"].values)

    logger.info(f'Calculating optimized top-k')
    global_results, _ = run_top_k_on_test(config, input_df, nb_docs,
                                          data_cleaner, data_formatter, model,
                                          to_csv_file=os.path.join(save_dir, 'optim_test_results.csv'))
    logger.info(f'Results: {global_results}')
    global_results['saved_model'] =  best_weights
    global_results['alpha_tree_mod'] = config['post_process']['alpha_tree_mod']
    global_results['beta_str_sim_mod'] = config['post_process']['beta_str_sim_mod']
    with open(os.path.join(save_dir, f'optim_top_k.json'), 'w') as f:
        json.dump(global_results, f)

    # push on minio
    if sync_with_minio:
        push_to_minio(save_dir)