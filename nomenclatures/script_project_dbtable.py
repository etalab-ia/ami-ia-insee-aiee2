#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# project_dbtable.py model_dir input_table output_table
#     model_dir : dossier du modèle à utiliser.
#     input_table : table postgreSQL de BI à projeter
#     output_table: table à créer pour verser les résultats
#
# Crée une table dans laquelle pour chaque BI, indexés par cabbi, sont stocké: (on prend comme ex la nomenclature pcs)
#    - cabbi
#    - pcs_proj : array contenant la projection du cabbi
#    - pcs_code_i (i dans [0, 9]): top-k des codes prédits
#    - pcs_score_i : score de prédiction associée
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

from configuration import *

from data_import.bdd import PostGre_SQL_DB
from data_import.csv_to_postgresql import read_table_description_file, import_csv_to_postegrsql
from training_utils import load_config, get_best_savedmodel
from script_run_top_k import load_model_from_save_dir, run_top_k_on_test


if __name__ == "__main__":

    """
    Script permettant de calculer les projections d'une table pour un modèle donné
    et de stocker le résultat dans une nouvelle table

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
    
    parser = argparse.ArgumentParser(
        description="Script pour calculer les projections des éléments d'une table")
    parser.add_argument("model_dir", type=str,
                        help="dossier du modèle entraîné")
    parser.add_argument("input_table", type=str,
                        help="nom de la table à projeter")
    parser.add_argument("output_table", type=str,
                        help="nom de la table à écrire")
    args = parser.parse_args()
    

    save_dir = os.path.abspath(args.model_dir)
    best_weights = get_best_savedmodel(save_dir)

    input_dbtable_name = args.input_table
    output_dbtable_name = args.output_table

    # add log in training dir
    test_log_file = os.path.join(save_dir, 'projection_dbtable.log')
    formatter = logging.Formatter(log_config['formatters']['simple']['format'])
    ch = logging.FileHandler(test_log_file)
    ch.setLevel('DEBUG')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger = logging.getLogger('ProjectionDbtable')
    
    logger.info(f'Running projection and top-k pour {save_dir}/{best_weights}')
    config = load_config(save_dir)

    # re-load all elements
    logger.info('Loading all trained objects')
    bdd = PostGre_SQL_DB(host=db_host, port=db_port, dbname=db_dbname, user=db_user, password=db_password)
    try:
        nomenclature, data_cleaner, data_formatter, model, top1_classifier = load_model_from_save_dir(save_dir, best_weights, bdd=bdd)
        modified = False
        if nomenclature.projections is None:
            raise Exception('No nomenclature projection found. Please run run_top_k.py on model dir')
        if nomenclature.ngram_hot_encodings is None:
            raise Exception('No nomenclature trigram repr found. Please run run_top_k.py on model dir')

    except Exception as e:
        logger.error(f'Error loading objects : {e}')
        exit(-1)

    # loading test data
    logger.info('Loading list of BIs')
    try:
        sql_request = config['data']['postgres_sql']
        fields = sql_request[sql_request.lower().index('select') + 7:sql_request.lower().index('from')].split(',')
        fields = [f.strip() for f in fields if '_c' not in f]
        sql_select_clause = 'SELECT ' + ','.join(fields) + ' FROM '
        sql_select_clause += input_dbtable_name + " "
        sql_count_request = f"SELECT COUNT(*) FROM {input_dbtable_name} "
        if "where" in sql_request.lower():
            sql_select_clause += sql_request[sql_request.lower().index('where'):]
            sql_count_request += sql_request[sql_request.lower().index('where'):]
        # sql_select_clause += 'limit 10'
    except Exception as e:
        logger.error(f'Error loading list of test BIs : {e}')
        exit(-1)

    # calculate values
    #bdd = PostGre_SQL_DB()
    bdd = PostGre_SQL_DB(host=db_host, port=db_port, dbname=db_dbname, user=db_user, password=db_password)
    batch_size = config['trainings']['training_params']['batch_size']
    input_df = bdd.read_from_sql_with_chunksize(sql_select_clause, batch_size)
    nb_docs = bdd.read_from_sql(sql_count_request).values[0][0]
    logger.info('Calculating projections')
    config['trainings']['data']['gt_column'] = None
    global_results, doc_results_df = run_top_k_on_test(config, input_df, nb_docs,
                                                       data_cleaner, data_formatter, model)
    if top1_classifier is not None:
        doc_results_df['top1_validation'] = top1_classifier.validate_top1(doc_results_df['top_k_similarities'])

    # update db
    model_name = config['data']['nomenclature']['name']
    table_desc_data = [
        ["cabbi","object","VARCHAR(90)","NOT NULL PRIMARY KEY"],
        [f"{model_name}_proj", "[float64]",f"double precision[]",""]
    ]
    for i in range(10):
        table_desc_data.append([f"{model_name}_code_{i}", "object", "VARCHAR(20)", ""])
        table_desc_data.append([f"{model_name}_score_{i}", "float64","double precision", ""])
    if top1_classifier is not None:
        table_desc_data.append([f"codage_auto", "bool", "boolean", ""])
    result_table_desc = pd.DataFrame(table_desc_data,
                                     columns=["FIELD_NAME","PANDAS_TYPE","SQL_TYPE","SQL_CONSTRAINT"])

    results_table = pd.DataFrame()
    results_table[['cabbi', f"{model_name}_proj"]] = doc_results_df[['cabbi', 'projection']]
    results_table[[f"{model_name}_code_{i}" for i in range(10)]] = \
        pd.DataFrame(doc_results_df['top_k_codes'].tolist())
    results_table[[f"{model_name}_score_{i}" for i in range(10)]] = \
        pd.DataFrame(doc_results_df['top_k_similarities'].tolist())
    if top1_classifier is not None:
        results_table['codage_auto'] = doc_results_df['top1_validation']

    path_to_desc_file = os.path.join(save_dir, f'{input_dbtable_name}_desc.csv')
    result_table_desc.to_csv(path_to_desc_file)
    path_to_results_file = os.path.join(save_dir, f'{input_dbtable_name}_results.csv')
    results_table.to_csv(path_to_results_file)

    psql_table_desc = read_table_description_file(output_dbtable_name, path_to_desc_file)
    import_csv_to_postegrsql(output_dbtable_name, psql_table_desc['sql'], 
                             ',', ';', path_to_results_file, psql_table_desc['dtype'])
