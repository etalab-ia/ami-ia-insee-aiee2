#!/usr/bin/env python3
"""
Script qui charge les données dans les index ES

Besoin des fichier de mapping "*settings*.json"

Données chargées :
    - ssplab/aiee2/data/Sirus/geo/geo-sirusYY_*.csv avec YY l'année
    - table sirus_{ANNEE} de la BDD

index rempli : sirus_{ANNEE}_e

2020/11/25
"""

# !pip install elasticsearch
import sys
sys.path.append("..")
from configuration import *
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

import data_import.bdd as bdd
import elastic as elastic


if __name__ == '__main__':
    
    #######################
    #   Variables du script
    #######################
    sirus_dbtable = "sirus_"+recap_year
    sirus_elastic_index = "sirus_"+recap_year+"_e"
    ES_index_must_be_created = True
    remote_data_dir="s3://ssplab/aiee2/data/Sirus/geo/"
    remote_geo_file_format= "geo-sirus"+recap_year[2:]

    #################
    #   Run
    #################
    my_driver = bdd.PostGre_SQL_DB(host=db_host, port=db_port, dbname=db_dbname, user=db_user, password=db_password)
    my_driver_es = elastic.ElasticDriver(host=elastic_host, port=elastic_port)

    list_file = fs.ls(remote_data_dir)
    list_file_filtered = []
    for file in list_file:
        if remote_geo_file_format in file:
            list_file_filtered.append(file)
    li = []
    
    for csv_file in list_file_filtered:
        with fs.open(csv_file) as f:
            df = pd.read_csv(f, sep=";",index_col=None, header=0, dtype=str)
        li.append(df)

    full_geo_data = pd.concat(li, axis=0, ignore_index=True)
    

    sql = f"SELECT * FROM {sirus_dbtable}"
    current_chunk = 0
    for df in my_driver.read_from_sql_with_chunksize(sql, chunksize=100000):
        logging.info(f'Processing chunk {current_chunk}')
        df = df.fillna('')
        df = df.astype(str)
        df['siret_id'] = df['sirus_id'] + df['nic']
        df = df.merge(full_geo_data, how='left', left_on='siret_id', right_on='siret')
        df['location'] = df['latitude'] +", " + df['longitude']
        df = df.fillna('')
        
        settings = my_driver_es.load_query("settings")
        fail_if_index_exists = False
        if current_chunk == 0:
            fail_if_index_exists = ES_index_must_be_created
        my_driver_es.export_data_to_es(settings, df, sirus_elastic_index,
                                       create_index=ES_index_must_be_created,
                                       fail_if_index_exists=fail_if_index_exists)
        current_chunk += 1
    
    logging.info('Done')