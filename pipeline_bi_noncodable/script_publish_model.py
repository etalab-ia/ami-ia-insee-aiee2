#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# 
# 
#
# @author Brivaël Sanchez
# @date: 2020/21/12

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

from script_runtime import load_pipeline_and_model


if __name__ == "__main__":
    """
    Script permettant de publier (modèle séléctionné pour la production) un modèle entraîné vers un dossier minio

    """
    parser = argparse.ArgumentParser(description="Script pour publier un modèle entraîné")
    
    parser.add_argument("remote_publishing_dir", type=str, help = "dossier minio où les modèles publiés sont stockés")
    parser.add_argument("model_dir", nargs='?', type=str, default=None, help = "dossier du modèle entraîné")
    args = parser.parse_args()

    with open(os.path.join(os.path.dirname(__file__), 'logging.conf.yaml'), 'r') as stream:
        log_config = yaml.load(stream, Loader=yaml.FullLoader)
    logging.config.dictConfig(log_config)
    logger = logging.getLogger()

    TRAININGS_LOCAL_DIR: str = "output/"
    minio_save_name = f"bi_noncodable_{date.today().isoformat()}"
    minio_save_path = os.path.join(args.remote_publishing_dir, minio_save_name)
    
    if hasattr(args, 'model_dir') and  args.model_dir:
        model_dir = os.path.abspath(args.model_dir)
    else:
        model_dir = TRAININGS_LOCAL_DIR
    conf_file = os.path.join(model_dir, 'config.yaml')
        
    # shutil.copyfile(conf_file, os.path.join(TRAININGS_LOCAL_DIR, "config.yaml"))

    try:
        logger.info('Trying to load saved elements')
        configs, cleaners, processes, model = load_pipeline_and_model(conf_file)
        if hasattr(model, 'load_model'):
            model.load_model(os.path.join(model_dir, 'model.h5'),
                             os.path.join(model_dir, 'dict_info.p'))
    except Exception as e:
        logger.error(f'Error loading objects : {e}')
        exit(-1)

    del cleaners,
    del processes
    del model

    logger.info('Creating temp directory')
    temp_dir = 'temp_publication'
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir)
    shutil.copyfile(os.path.join(model_dir, "config.yaml"), os.path.join(temp_dir, "config.yaml"))
    shutil.copyfile(os.path.join(model_dir, "dict_info.p"),
                    os.path.join(temp_dir, "dict_info.p"))
    shutil.copyfile(os.path.join(model_dir, "tokenizer.p"),
                    os.path.join(temp_dir, "tokenizer.p"))
#     os.makedirs(os.path.join(temp_dir, 'model'))
    shutil.copyfile(os.path.join(model_dir, 'model.h5'),
                    os.path.join(temp_dir, 'model.h5'))

    try:
        logger.info('Verify copied elements')
        configs_model, cleaners, processes, model = load_pipeline_and_model(os.path.join(temp_dir, 'config.yaml'))
        if hasattr(model, 'load_model'):
            model.load_model(os.path.join(temp_dir, 'model.h5'),
                             os.path.join(temp_dir, 'dict_info.p'))
    except Exception as e:
        logger.error(f'Error loading objects : {e}')
        exit(-1)

    minio_save_path : os.path.join(minio_save_path, minio_save_name) + "/"
    logger.info(f'Pushing to {minio_save_path}')
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': configs['minio']['endpoint']})
    fs.put(temp_dir, minio_save_path, recursive=True)
    logger.info(f'Done!')
    
    shutil.rmtree(temp_dir)