#!/usr/bin/env python3
"""
2021/01/19

Publie un model sur minio, se base sur le dossier output

Auteur: Brivaël Sanchez
"""
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
import argparse
from script_runtime import *

if __name__ == "__main__":
    """
    Script permettant de publier (modèle séléctionné pour la production) un modèle entraîné vers un dossier minio

    """
    with open(os.path.join(os.path.dirname(__file__), 'logging.conf.yaml'), 'r') as stream:
        log_config = yaml.load(stream, Loader=yaml.FullLoader)
    logging.config.dictConfig(log_config)
    logger = logging.getLogger()


    parser = argparse.ArgumentParser(description="Script pour publier un modèle entraîné")
    parser.add_argument("remote_publishing_dir", type=str, help = "dossier minio où les modèles publiés sont stockés")
    parser.add_argument("model_dir", type=str, help = "dossier du modèle entraîné")
    args = parser.parse_args()
        
    model_dir = os.path.abspath(args.model_dir)
    minio_save_name = f"siret_bi_{date.today().isoformat()}"
    minio_save_path = os.path.join(args.remote_publishing_dir, minio_save_name)

    try:
        configs, cleaners, processes, model, meta_model, threshold = load_pipeline_and_model(os.path.join(model_dir, "config_bi.yaml"))
    except Exception as e:
        logger.error(f'Error loading objects : {e}')
        exit(-1)

    del cleaners,
    del processes
    del model

    logger.info('Creating temp directory')
    temp_dir = 'temp_publication'
    os.makedirs(temp_dir)
    shutil.copyfile(os.path.join(model_dir, "config_bi.yaml"),
                    os.path.join(temp_dir, "config.yaml"))
    shutil.copyfile(os.path.join(model_dir, "config_si.yaml"),
                    os.path.join(temp_dir, "config_si.yaml"))
    shutil.copyfile(os.path.join(model_dir, "paths.json"),
                    os.path.join(temp_dir, "paths.json"))
    shutil.copyfile(os.path.join(model_dir, "dict_info.pickle"),
                    os.path.join(temp_dir, "dict_info.pickle"))
    shutil.copyfile(os.path.join(model_dir, "tokenizer.p"),
                    os.path.join(temp_dir, "tokenizer.p"))
    shutil.copyfile(os.path.join(model_dir, "meta_model.p"),
                    os.path.join(temp_dir, "meta_model.p"))
    shutil.copyfile(os.path.join(model_dir, "threshold.p"),
                    os.path.join(temp_dir, "threshold.p"))
    shutil.copyfile(os.path.join(model_dir, "best_param_optuna.json"),
                    os.path.join(temp_dir, "best_param_optuna.json"))
#     os.makedirs(os.path.join(temp_dir, 'model'))
    shutil.copytree(os.path.join(model_dir, 'model_0'),
                    os.path.join(temp_dir, 'model'))

    try:
        configs, cleaners, processes, model, meta_model, threshold = load_pipeline_and_model(os.path.join(temp_dir, 'config.yaml'))
    except Exception as e:
        logger.error(f'Error loading objects : {e}')
        exit(-1)

    del cleaners,
    del processes
    del model

    logger.info(f"Pushing to {configs['minio']['endpoint']}/{minio_save_path}")
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': configs['minio']['endpoint']})
    fs.put(temp_dir, minio_save_path, recursive=True)
    logger.info(f'Done!')
    
    shutil.rmtree(temp_dir)
    
    
    