# coding: utf8
"""
Script d'import CSV -> PostgreSQL

Author : bsanchez@starclay.fr
date : 16/07/2020
"""
import sys
sys.path.append("..")
import getopt
import os
import s3fs, botocore
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import psycopg2
import time
from tqdm import tqdm
import io
import sys
import data_import.bdd as bdd
import yaml
from ast import literal_eval
import csv
from config import *
import logging
import logging.config


logger = logging.getLogger('ImportCsvToPostgresql')

def apply_dtype(df:pd.DataFrame, dtypes):
    """
    Applique les dtypes d'un dictionnaire sur un dataframe
    
    :param df: dataframe 
    :param dtypes: dictionnaire ayant pour clé les colonnes et pour valeur les dtypes pandas
    
    :returns: df ayant les dtypes modifié
    """
    for col in dtypes.keys():
        try:
            if col.lower() not in df:
                df[col.lower()] = ""
            if (dtypes[col] == "Int64") or (dtypes[col] == "Int32") :
                df[col.lower()] = df[col.lower()].astype('float').astype(dtypes[col])
            else:
                df[col.lower()] = df[col.lower()].astype(str)
        except Exception as e:
            logger.error(F'import_csv_to_postegrsql.apply_dtype : {col} - {e.__class__.__name__} - {e}')
            raise e
    return df


def create_table_description(table_name, df_table_desc):
    """
    Crée la description de la table

    :param table_name: str, nom de la table
    :param df_table_desc: dataframe (voir fichier typage_*.csv)
    :return: dict {'sql': requete SQL créant la table,'dtype': {field_name: pandas_type}}
    """
    df_table_desc.fillna('',inplace=True)
    dictionnary = pd.Series(df_table_desc.PANDAS_TYPE.values,index=df_table_desc.FIELD_NAME.str.lower()).to_dict()
    sql = F'CREATE TABLE {table_name} ('
    for index, row in df_table_desc.iterrows():
        if index != 0:
            sql += ','
        sql += F"{row['FIELD_NAME'].lower()} {row['SQL_TYPE']} {row['SQL_CONSTRAINT']}"
    sql += ');'

    description = {'sql':sql,'dtype':dictionnary}
    return description

def read_table_description_file(table_name, file_path):
    """
    Lit un fichier de description de table à créer et renvoie un dictionnaire de types pandas
    et une commande SQL de création de la table

    :param table_name: str, nom de la table
    :param file_path: chemin du fichier à charger
    :returns:
        - dict : {"nom_de_champ":"type_pandas"}
        - string: commande SQL
    """
    return create_table_description(table_name, pd.read_csv(file_path,dtype=str))

def import_csv_to_postegrsql(table_name, create_table_query, 
                             separator_csv, separator_buffer, fichier, dtype_cols, 
                             force_import=False):
    """
    Importe un csv vers la base en créant/écrasant la table :param table_name:.

    :param table_name: nom de la table
    :param create_table_query: requete sql de création de la table
    :param separator_csv: separateur pour parser le csv
    :param separator_buffer: separateur pour mettre le csv dans le buffer d'import
    :param fichier: csv à importer
    :param dtype_cols: datatype pandas du csv
    :param force_import: Si vrai, écrase une table portant le même nom 
    :returns: void
    """
    my_driver = bdd.PostGre_SQL_DB(host=db_host, port=db_port, dbname=db_dbname, user=db_user, password=db_password,logger=logger)   # errors already in the logs
    chunk_size = 3000 
    
    try:
        conn = psycopg2.connect(F"host='{my_driver.HOST}' port='{my_driver.PORT}' dbname='{my_driver.DBNAME}' user='{my_driver.USER}' password='{my_driver.PASSWORD}'")
        cur = conn.cursor()
        if not force_import:
            cur.execute(F"SELECT * FROM information_schema.tables WHERE table_name = '{table_name}'")
            exist = bool(cur.rowcount)
            if exist:
                logger.error(F"Une table ayant le même nom que {table_name} a été trouvé, echec de l'import")
                raise ValueError(F"Une table ayant le même nom que {table_name} a été trouvé, echec de l'import")

        cur.execute(F"DROP TABLE IF EXISTS {table_name}")
        cur.execute(create_table_query)
        arrays_to_load = []
        for k, v in dtype_cols.items():
            if v[0] == '[' and v[-1] == "]":
                arrays_to_load.append(k)
        for k in arrays_to_load:
            dtype_cols[k] = "object"

        for df in tqdm(pd.read_csv(fichier,sep=separator_csv,chunksize=chunk_size,dtype=dtype_cols, engine='python',encoding='utf-8')):
            for k in arrays_to_load:
                df[k] = df[k].apply(lambda s: s.replace('[','{').replace(']', '}'))
            df.columns = [x.lower() for x in df.columns]
            df.dropna(axis=0, subset=['cabbi'], inplace=True)
            df = df.fillna('')
            df = apply_dtype(df,dtype_cols)
            df = df[dtype_cols.keys()]
            buffer = io.StringIO()
            df.to_csv(buffer, index=False, header=False, sep=separator_buffer)
            buffer.seek(0)
            cur.copy_from(buffer, F'{table_name}', sep=separator_buffer, null='')
            conn.commit()
        conn.close()
    except psycopg2.Error as e:
        logger.error(F'import_csv_to_postegrsql : {e.__class__.__name__} - {e}')
        raise e



if __name__ == "__main__":
    
    #getopt
    full_cmd_arguments = sys.argv
    argument_list = full_cmd_arguments[1:]
    short_options = "c:fh"
    long_options = ["config", "force", "help"]

    def help():
        print('**** CSV_TO_POSTGRESQL *****')
        print("Ce script permet l'import de données depuis des fichiers CSV sur minio vers une base postgreSQL")
        print("Voir Readme")
        print("Options supportées :")
        print("  -c   --config     préciser le fichier de config de l'opération (défaut: config_import.yaml)")
        print("  -f   --force      Forcer la réinsertion des données from scratch si la table existe déjà")
        print("  -h   --help       afficher l'aide")

    try:
        arguments, values = getopt.getopt(argument_list, short_options, long_options)
    except getopt.error as err:
        logger.error(F"{err.__class__.__name__} - {err}")
        sys.exit(2)
        
    #default values for params
    conf_file = os.path.join(os.path.dirname(__file__), 'config_import.yaml')
    force_import = False
    
    for current_argument, current_value in arguments:
        if current_argument in ("-c", "--config"):
            if len(current_value):
                conf_file = current_value
        if current_argument in ("-f", "--force"):
            force_import = True
        if current_argument in ("-h", "--help"):
            help()
            sys.exit(0)
            
    # open config
    with open(conf_file) as f:
        try:
            configs = yaml.safe_load(f) 
        except ValueError:
            logger.error(F"Erreur dans le chargement du fichier de configuration {conf_file}") 
            sys.exit(2)

    # import data
    import_errors = []

    for file_desc in configs['files']:
        try:
            descr = read_table_description_file(file_desc['table_name'], 
                                                os.path.join(os.path.dirname(__file__), file_desc['typage_file']))
        except Exception as e:
            logger.error(F"{e.__class__.__name__} - {e}")
            import_errors.append(file_desc['minio_url'])
            continue
        query_create_table = descr['sql']
        dtypes = descr['dtype']
        try:
            with fs.open(file_desc['minio_url']) as f:
                logger.info(F"Importation du csv {file_desc['table_name']}")
                import_csv_to_postegrsql(file_desc['table_name']
                                ,query_create_table
                                ,file_desc['options_csv_dta']['separator_csv']
                                ,file_desc['options_csv_dta']['separator_buffer']
                                ,f
                                ,dtypes,
                                force_import=force_import)
        except botocore.exceptions.NoCredentialsError as e:
            logger.error(F"{e.__class__.__name__} - {e}")
            exit(2)
        except Exception as e:
            if isinstance(e, FileNotFoundError):
                logger.error(F"{e.__class__.__name__} - {file_desc['minio_url']}")
            # errors are already in the logs
            import_errors.append(F"{e.__class__.__name__} - {file_desc['minio_url']}")
            continue
    
    if import_errors:

        logger.error('Could not import following files (check logs for errors) :' + ",".join(import_errors))
        sys.exit(2)

    sys.exit(0)