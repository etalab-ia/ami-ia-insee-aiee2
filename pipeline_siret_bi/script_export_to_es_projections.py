#!/usr/bin/env python3
"""
Script qui charge les données dans les index ES

Besoin des fichier de mapping "*settings*.json"

données chargées : table sirus_projection de la BDD
index créé : sirus_2020_projection

2020/11/25
"""

# !pip install elasticsearch
import sys
sys.path.append("..")
import csv
import yaml
from elasticsearch import Elasticsearch
from elasticsearch.helpers import parallel_bulk
import s3fs
import time
import os
from collections import deque
import json
import pandas as pd
import logging
import logging.config

from config import *
import data_import.bdd as bdd
import elastic as elastic


if __name__ == '__main__':
    with open(os.path.join(os.path.dirname(__file__), 'logging.conf.yaml'), 'r') as stream:
        log_config = yaml.load(stream, Loader=yaml.FullLoader)
    logging.config.dictConfig(log_config)
    
    #######################
    #   Variables du script
    #######################
    sirus_proj_table = "sirus_projections_"+recap_year
    sirus_elastic_index = "sirus_"+recap_year+"_projections"
    ES_index_must_be_created = True

    #################
    #   Run
    #################
    my_driver = bdd.PostGre_SQL_DB(host=db_host, port=db_port, dbname=db_dbname, user=db_user, password=db_password)
    my_driver_es = elastic.ElasticDriver(host=elastic_host, port=elastic_port)
    
    sql = f"SELECT * FROM {sirus_proj_table}"
    current_chunk = 0
    for df in my_driver.read_from_sql_with_chunksize(sql, chunksize=100000):
        logging.info(f'Processing chunk {current_chunk}')
        df = df.fillna('')
        df = df.astype(str)
        df['siret'] = df['sirus_id'] + df['nic']
        df = df.fillna('')
        
        settings = my_driver_es.load_query("settings_sirus_projection")
        fail_if_index_exists = False
        if current_chunk == 0:
            fail_if_index_exists = ES_index_must_be_created
        my_driver_es.export_data_to_es(settings, df, sirus_elastic_index,
                                       create_index=ES_index_must_be_created,
                                       fail_if_index_exists=fail_if_index_exists)
        current_chunk += 1
    
    logging.info('Done')
