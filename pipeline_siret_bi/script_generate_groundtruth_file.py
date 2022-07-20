#!/usr/bin/env python3
"""
Fichier qui génère un dataframe contenant tout les BI 
avec les sirus_id/nic correspondant.

Est nécessaire pour script_runtime, meta_model_optimisation, script_generate_scoring

Année 2019
"""
import sys
sys.path.append("..")
import os
import yaml
import logging
import logging.config
import pandas as pd
import pickle
import data_import.bdd as bdd


if __name__ == '__main__':
    with open(os.path.join(os.path.dirname(__file__), 'logging.conf.yaml'), 'r') as stream:
        log_config = yaml.load(stream, Loader=yaml.FullLoader)
    logging.config.dictConfig(log_config)

    #######################
    #   Variables du script
    #######################
    work_dir = 'trainings/2021-02-18_25'
    rp_table = 'rp_final_2019'
    sirus_table = 'sirus_2019'
    vagues_lots_table = 'bi_lot_2019'

    #################
    #   Run
    #################
    my_driver = bdd.PostGre_SQL_DB()
    # Generate eval file
    output_file = os.path.join(work_dir, "df_solution_eval.p")
    logging.info(f'Getting eval GT data to {output_file}')
    with open(os.path.join(work_dir, "cabbi_val.p"), "rb") as f:
        cabbis_eval = pickle.load(f)
    cabbis_list = "'" + "','".join(cabbis_eval) + "'"
    sql_bi = f"""SELECT r.cabbi, b.lot_id, b.vague_id, r.siretc, s.nic, s.sirus_id FROM {rp_table} r 
                 INNER JOIN {sirus_table} s ON (LEFT(r.siretc,9) = s.sirus_id) AND s.nic = RIGHT(r.siretc, 5) 
                 LEFT JOIN {vagues_lots_table} b ON r.cabbi = b.cabbi 
                 WHERE r.cabbi IN ({cabbis_list}) AND siretc IS NOT NULL LIMIT 100000;"""
    
    df = my_driver.read_from_sql(sql_bi)
    with open(output_file, "wb") as output_file:
        pickle.dump(df, output_file)

    # Generate eval file
    output_file = os.path.join(work_dir, "df_solution_test.p")
    logging.info(f'Getting test GT data to {output_file}')
    sql_bi = f"""SELECT r.cabbi, r.siretc, s.nic, s.sirus_id FROM {rp_table} r 
                 INNER JOIN {sirus_table} s ON (LEFT(r.siretc,9) = s.sirus_id AND s.nic = RIGHT(r.siretc, 5)) 
                 LEFT JOIN {vagues_lots_table} b ON r.cabbi = b.cabbi 
                 WHERE b.cabbi IS NULL LIMIT 100;"""
    
    df = my_driver.read_from_sql(sql_bi)
    with open(output_file, "wb") as output_file:
        pickle.dump(df, output_file)

