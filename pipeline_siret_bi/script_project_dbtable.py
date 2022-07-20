#!/usr/bin/env python3
"""
Ce script permet de calculer les projections Sirus ou BI et de les mettre dans la DB

"""
import sys
sys.path.append("..")
import os
import os.path
from os import path

import shutil
import glob
import yaml
import getopt
import pickle
import json
from datetime import date
import s3fs
import logging

import numpy as np
import pandas as pd
from numpy import load

import sqlalchemy
import sqlalchemy.dialects.postgresql

import data_import.bdd as bdd
from training_classes.utils import *
from training_classes.CleanerRemoveSpecialChar import CleanerRemoveSpecialChar
from training_classes.CleanerAdrCltX import CleanerAdrCltX
from training_classes.CleanerAdrDltX import CleanerAdrDltX
from training_classes.CleanerConcat_profs import CleanerConcat_profs
from training_classes.ProcessMLPSiamese import ProcessMLPSiamese
from training_classes.ModelSimilarity import MLPSimilarity
from training_classes import preprocessing as pre
from script_runtime import load_pipeline_and_model, project

     
def export_projections_to_bdd(projection_dir: str,
                              bdd_table="sirus_projection"):
    """
    Export des embedding dans une bdd
    
    :params projection_dir: dossier contenant les projections (ids.p et matrix_embedding.p)
    :param bdd_table: table à remplir
    """
    with open(os.path.join(projection_dir, "ids.p"), "rb") as input_file:
        sirets = pickle.load(input_file)
    list_file_si = glob.glob(f"{projection_dir}/matrix*.p")
    list_file_si = sorted(list_file_si, key=str.lower)

    for i, file_si in enumerate(list_file_si):
        if i == 0:
            with open(file_si, "rb") as input_file_si:
                matrix_embedding_si = pickle.load(input_file_si)
            full_si = matrix_embedding_si
        else:
            with open(file_si, "rb") as input_file_si:
                matrix_embedding_si = pickle.load(input_file_si)
            matrix_embedding_si = matrix_embedding_si
            full_si = np.vstack([full_si, matrix_embedding_si])
    
    df_sirus_emb = pd.DataFrame()
    df_sirus_emb['sirus_id'] = sirets['sirus_id']
    df_sirus_emb['nic'] = sirets['nic']
        
    df_sirus_emb['embdedding'] = full_si.tolist()
    
    my_driver = bdd.PostGre_SQL_DB(host=db_host, port=db_port, dbname=db_dbname, user=db_user, password=db_password)
    ### VIRER LA LIGNE DU DESSUS UN JOUR ?
    df_sirus_emb.to_sql(bdd_table
                        , my_driver.engine
                        , if_exists = 'append'
                        , index = False 
                        ,  dtype = {  'sirus_id' : sqlalchemy.String()
                                    , 'nic' : sqlalchemy.String()
                                    , 'embdedding' : sqlalchemy.dialects.postgresql.ARRAY(sqlalchemy.types.REAL)}
                       )

        
if __name__ == '__main__':
    with open(os.path.join(os.path.dirname(__file__), 'logging.conf.yaml'), 'r') as stream:
        log_config = yaml.load(stream, Loader=yaml.FullLoader)
    logging.config.dictConfig(log_config)
    logger = logging.getLogger('runtime')
    logger.info("Commencement script_projection")

    #######################
    #   Variables du script
    #######################
    model_dir = 'import_model'
    conf_file = os.path.join(model_dir, 'config_si.yaml')
    
    work_dir = os.path.join(model_dir, 'runtime')
    BI_EMB_DIRECTORY = os.path.join(work_dir, 'emb_bi')
    EMB_TABLE = "sirus_proj_test_2020"

    #################
    #   Run
    #################
    # load config
    current_dir = os.path.dirname(__file__)
    conf_file = os.path.join(current_dir, conf_file)
    with open(conf_file) as f:
        configs = yaml.safe_load(f)
    my_driver = bdd.PostGre_SQL_DB(host=db_host, port=db_port, dbname=db_dbname, user=db_user, password=db_password)
    mydir = os.path.dirname(os.path.realpath(__file__))
    sql_sirus = configs['import_data_sql'] 
    
    # load model
    logging.info(f"Chargement du modèle SIRET depuis {conf_file}")
    configs, cleaners, processes, model, _, _ = load_pipeline_and_model(conf_file, for_training=False)
    logging.info(f"Modèle chargé")
#     my_driver.delete_table("sirus_projection")
    
    current_df_ind = 0
    work_dir = os.path.dirname(os.path.realpath(__file__))

    for df in my_driver.read_from_sql_with_chunksize(sql_sirus, chunksize=100000):
        logging.info(f'Processing chunk {current_df_ind}')
        shutil.rmtree(BI_EMB_DIRECTORY)
        os.makedirs(BI_EMB_DIRECTORY)
        # calculate projections
        project(df, 
                configs, cleaners, processes, model, 
                projections_dir=BI_EMB_DIRECTORY, work_dir=os.path.join(work_dir, 'tmp'))
        # push in db
        export_projections_to_bdd(BI_EMB_DIRECTORY, EMB_TABLE)
        current_df_ind += 1
