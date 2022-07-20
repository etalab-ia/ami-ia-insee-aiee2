#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# publish_model.py remote_publishing_dir [model_dir]
#     model_dir : dossier du modèle à publier. 
#                    Défault : training le plus récent dans config['local']['trainings_dir']/config['data']['nomenclature']['name']
#     remote_publishing_dir : chemin minio vers le dossier de publication
#                    Le modèle sera publié dans remote_publishing_dir/NOMNAME_YYYY_MM_JJ
#
# utilise config.yaml dans le dossier de training
#
# @author cyril.poulet@starclay.fr
# @date: nov 2020


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
from datetime import date

from data_import.bdd import PostGre_SQL_DB
from data_import.csv_to_postgresql import read_table_description_file, import_csv_to_postegrsql
from training_utils import load_config, get_best_savedmodel, get_trainings_dir, get_last_local_training_dir
from script_run_top_k import load_model_from_save_dir


if __name__ == "__main__":

    """
    Script permettant de publier un modèle entraîné vers un dossier minio

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
        description="Script pour publier un modèle entraîné")
    parser.add_argument("remote_publishing_dir", type=str,
                    help="dossier minio où les modèles publiés sont stockés")
    parser.add_argument("model_dir", nargs='?', type=str, default=None,
                        help="dossier du modèle entraîné")
    args = parser.parse_args()

    if hasattr(args, 'model_dir') and  args.model_dir:
        model_dir = os.path.abspath(args.model_dir)
    else:
        model_dir = get_last_local_training_dir(TRAININGS_LOCAL_DIR)

    best_weights = get_best_savedmodel(model_dir)
    config = load_config(model_dir)

    minio_save_name = f"{config['data']['nomenclature']['name']}_{date.today().isoformat()}"
    minio_save_path = os.path.join(args.remote_publishing_dir, minio_save_name)

    # add log in training dir
    test_log_file = os.path.join(model_dir, 'model_publication.log')
    formatter = logging.Formatter(log_config['formatters']['simple']['format'])
    ch = logging.FileHandler(test_log_file)
    ch.setLevel('DEBUG')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger = logging.getLogger('ModelPublication')
    
    logger.info(f'publishing {model_dir} to {minio_save_path}')
    

    logger.info('Loading all trained objects')
    try:
        nomenclature, data_cleaner, data_formatter, model, top1_classifier = load_model_from_save_dir(model_dir, best_weights)
        modified = False
        if nomenclature.projections is None:
            raise Exception('No nomenclature projection found. Please run run_top_k.py on model dir')
        if nomenclature.ngram_hot_encodings is None:
            raise Exception('No nomenclature trigram repr found. Please run run_top_k.py on model dir')

    except Exception as e:
        logger.error(f'Error loading objects : {e}')
        exit(-1)

    logger.info('Creating temp directory')
    temp_dir = 'temp_publication'
    os.makedirs(temp_dir)
    data_formatter.save(os.path.join(temp_dir, 'batcher'))
    with open(os.path.join(temp_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f)
    shutil.copyfile(os.path.join(model_dir, "paths.json"),
                    os.path.join(temp_dir, "paths.json"))
    os.makedirs(os.path.join(temp_dir, 'train_weights'))
    shutil.copytree(os.path.join(model_dir, 'train_weights', best_weights),
                    os.path.join(temp_dir, 'train_weights', best_weights))
    if top1_classifier is not None:
        shutil.copyfile(os.path.join(model_dir, config['top1_classifier']['scaler_file']),
                        os.path.join(temp_dir, config['top1_classifier']['scaler_file']))
        shutil.copyfile(os.path.join(model_dir, config['top1_classifier']['model_file']),
                        os.path.join(temp_dir, config['top1_classifier']['model_file']))


    logger.info('verifying correct load')
    del nomenclature, data_cleaner, data_formatter, model
    
    try:
        nomenclature, data_cleaner, data_formatter, model, top1_classifier = load_model_from_save_dir(temp_dir, best_weights)
    except Exception as e:
        logger.error(f'Error loading objects : {e}')
        exit(-1)
    
    logger.info('pushing to minio')
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': config['minio']['endpoint']})
    fs.put(temp_dir, minio_save_path, recursive=True)

    shutil.rmtree(temp_dir)
    logger.info('done')