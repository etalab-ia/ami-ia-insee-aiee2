#!/usr/bin/env python3
"""
2020/10/27

Script qui simule l'api de prediction:
 
BI -> projection
BI -> enrichissement -> topk (besoin de la projection) -> metriques

Auteur: Brivaël Sanchez
"""
import sys
sys.path.append("..")
import os
import os.path
from os import path
import shutil
import glob
import yaml
import json
import getopt
import pickle
from datetime import date
import random
import logging
import s3fs
from dataclasses import dataclass

import pandas as pd
import numpy as np
from scipy.spatial import distance
from collections import Counter

import sqlalchemy

import data_import.bdd as bdd
from training_classes import preprocessing as pre

from training_classes.CleanerRemoveSpecialChar import CleanerRemoveSpecialChar
from training_classes.CleanerAdrCltX import CleanerAdrCltX
from training_classes.CleanerAdrDltX import CleanerAdrDltX
from training_classes.CleanerConcat_profs import CleanerConcat_profs
from training_classes.ProcessMLPSiamese import ProcessMLPSiamese
from training_classes.ModelSimilarity import MLPSimilarity
from training_classes.utils import *
from script_metamodel_optimisation import *
from geocodage import *
import elastic as elastic

import asyncio
from config import *

@dataclass(frozen=True)
class Echo:
    siret: str
    from_request: str 
    score_request: str 
        
@dataclass(frozen=True)
class PerformanceES:
    siret: str
    cabbi: str 
    nb_fois_trouve: str

###################
#   Model loading
###################
def load_pipeline_and_model(config_file_in_model_dir, 
                            fs=None,
                            for_training=False):
    """
    Charge les classes du pipeline utilisé par le modèle
    
    :params config_file_in_model_dir: chemin du fichier de configuration dans le dossier de training
    :param fs: s3fs.Filesystem (si None, il sera créé avec les parametres par défaut)
    :param for_training: bool. Si False, renvoie une erreur si pas de meta-modèle trouvé
    :return configs, cleaners, processes, model, meta_model, threshold: classe du preprocessing du model et le model
    """
    
    # Init meta variable
    thismodule = sys.modules[__name__]

    model_dir = os.path.dirname(config_file_in_model_dir)
    with open(config_file_in_model_dir) as f:
        configs = yaml.safe_load(f)
    if fs is None:
        fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': configs['minio']['endpoint']})

    ########
    # INIT
    # Instantiation des opération contenues dans le fichier de config avec leurs 
    # paramètres
    ########
    
    try:
        cleaners = []
        for cleaner in configs['cleaners']:
            for obj in cleaner.keys():
                class_ = getattr(thismodule, cleaner[obj]['type'])
                instance = class_(cleaner[obj]['cols'])
                cleaners.append(instance)   
    except ValueError as e:
        logging.error(F"Erreur dans la creation des cleaners: {str(e)}'")
        sys.exit(2)
    try:
        processes = []
        for transformer in configs['processes']:
            for obj in transformer.keys():
                class_ = getattr(thismodule, transformer[obj]['type'])
                attr = transformer[obj]['param']
                attr['local_path_run'] = model_dir
                instance = class_(**attr)
                processes.append(instance)
    except ValueError as e:
        logging.error(F"Erreur dans la creation des Processes: {str(e)}")
        sys.exit(2)
    try:
        list_model = []
        for model in configs['models']:
            for obj in model.keys():
                class_ = getattr(thismodule, model[obj]['type'])
                attr = model[obj]['param']
                attr['local_path_run'] = model_dir
                instance = class_(**attr)
                list_model.append(instance)
    except ValueError as e:
        logging.error(F"Erreur dans la creation des modeles: {str(e)}")
        sys.exit(2)
                
    model = list_model[0] # TODO:  On a forcément qu'un seul model
    model_to_load = os.path.join(model_dir, 'model_0')
    if not os.path.exists(model_to_load):
        model_to_load = os.path.join(model_dir, 'model')
    if os.path.exists(model_to_load):
        model.load_model(model_path=model_to_load, dict_info_path=model_dir)
    else:
        logging.warn('No trained model found')

    #Chargement des coeff d'optimisation
    if os.path.isfile(os.path.join(model_dir, "best_param_optuna.json")):
        with open(os.path.join(model_dir, "best_param_optuna.json"), "r") as input_file:
            best_trial = json.load(input_file)
        coeff_nb_fois_trouve = best_trial["coeff_nb_fois_trouve"]
        coeff_naf_score = best_trial["coeff_naf_score"]
    else:
        logging.warning(f"Fichier pickle des coeff d'optimisation inexistant")
        coeff_nb_fois_trouve = 0
        coeff_naf_score = 0
    configs['final_score_coeffs'] = {
        'coeff_naf_score': coeff_naf_score,
        'coeff_nb_fois_trouve': coeff_nb_fois_trouve
    }
    
    meta_model = None
    if os.path.isfile(os.path.join(model_dir, "meta_model.p")):
        with open(os.path.join(model_dir, "meta_model.p"), "rb") as input_file:
            meta_model = pickle.load(input_file)
    else:
        if not for_training:
            raise RuntimeError('No meta_model.p found')
    
    threshold = 0.5
    if os.path.isfile(os.path.join(model_dir, "threshold.p")):
        with open(os.path.join(model_dir, "threshold.p"), "rb") as input_file:
            threshold = pickle.load(input_file)

    return configs, cleaners, processes, model, meta_model, threshold
    

###################
#   Data projection
###################
def project(data, configs, cleaners, processes, model, 
            projections_dir, work_dir='tmp_dir'):
    """
    Projecte un jeu de donnée dans un espace de représentation.
    Nécéssite un model de similarité déjà entrainé

    :param data: df à projeter
    :param configs, cleaners, processes, model: modèle chargé
    :param projections_dir: dossier ou sauvegarder le résultat
    :param work_dir: dossier ou sauvegarder les fichiers intermédiaires
    """
    ###################
    # Overwrite / clean tmp file
    ###################
    try:
        os.makedirs(work_dir, exist_ok=True)
        # clean
        for f in glob.glob(os.path.join(work_dir, "*")):
            os.remove(f)
    except ValueError:
        print(F"Erreur dans la suppression des fichiers temporaires")
        logging.error(F"Erreur dans la suppression des fichiers temporaires")
        sys.exit(2)

    print("Writing data...")
    data_file = os.path.join(work_dir, "dataset.csv")
    data.to_csv(data_file, sep=';',index=False)

    if 'cols_id' in configs:
        with open(os.path.join(projections_dir, "ids.p"), "wb") as f:
            pickle.dump(data[configs['cols_id']], f)
    
    try:
        for cleaner in cleaners:
            list_of_files = glob.glob(f'{work_dir}/*') 
            input_file = max(list_of_files, key = os.path.getctime)
            if input_file == data_file:
                output_file = os.path.join(work_dir, "1.csv")
            else:
                output_file = os.path.join(work_dir, incrementTmpFile(input_file) + ".csv")
            cleaner.process(input_file, output_file)
    except ValueError:
        print(F"Erreur dans l'exécution de {cleaner.__class__.__name__}")
        logger.error(F"Erreur dans l'exécution de {cleaner.__class__.__name__}")
        sys.exit(2)

    for process in processes:
        list_of_files = glob.glob(f'{work_dir}/*') 
        input_file = max(list_of_files, key = os.path.getctime)
        output_file = os.path.join(work_dir, incrementTmpFile(input_file) + ".npy")
        process.run(input_file, output_file)

    list_of_files = glob.glob(f'{work_dir}/*') 
    latest_file = max(list_of_files, key=os.path.getctime)
    X = np.load(latest_file, allow_pickle=True)
    model.predict(X[0], projections_dir)
        
   
###################
#   Echo generation
###################
def data_enrichment(df, driver_elastic=None, elastic_index_bi='rp_2020_e'):
    """
    Enrichissement des donnée rp.
    On prend les cabbi, pour récupérer les donnée enrichie dans ES.
    Puis on les merge au dataframe
    
    :params df: daframe de bi
    :param driver_elastic: driver elastic instancié. Si None: un sera créé avec les params par défaut
    :param elastic_index_bi: index à interroger
    :returns: dataframe mergé
    """
    if driver_elastic is None:
        driver_elastic = elastic.ElasticDriver()
    var = [ {'CABBI':x}  for x in df.cabbi.values]
    res = driver_elastic.request("cabbi",elastic_index_bi, var)

    # Chaque dataframe dans la liste ne contient qu'une seule valeur étant donnée qu'on requete sur l'ID
    dfs: list = driver_elastic.result_to_dataframe(res)
    df_e = pd.concat(dfs,sort=True)
    if len(df_e) == 0:
        # no match
        return df
    df_merged = df.merge(df_e, how='left', left_on='cabbi', right_on='_id',validate='one_to_one')
    df_merged = df_merged.fillna("")
    return df_merged
    

def get_most_similar_elastic_bi(df, num_candidate = 10, 
                                driver_elastic=None, elastic_index_bi='rp_2020_e'):
    """
    On execute les requetes ES pour récupérer les BI les plus similaires pour récupéré leur siret
    
    :params df: dataframe de bi
    :params num_candidate: nb de siret réponse par requete
    :param driver_elastic: driver elastic instancié. Si None: un sera créé avec les params par défaut
    :param elastic_index_bi: index à interroger
    :returns: dict_resp dict pour clé cabbi et pour value list de bi 
    """
    dict_resp = {}
    
    list_cabbi = df.cabbi.values
    
    for cabbi in list_cabbi:
        dict_resp[cabbi] = []
    if driver_elastic is None:
        driver_elastic = elastic.ElasticDriver()
    
    def get_bi_siret(bi_dict):
        if 'siretm' in bi_dict and bi_dict['siretm']:
            return bi_dict['siretm']
        return bi_dict['siretc']
 
    chunksize = 50 
    list_df = [df[i:i+chunksize] for i in range(0,df.shape[0],chunksize)]
    for part_df in list_df:
        current_bi = part_df.cabbi.values
        var = []
        for idx, row in part_df.iterrows():

            var.append({  'rs_x': row['rs_x']
                        , 'clt_c_c': row['clt_c_c']
                        , 'nomvoi_x': row['nomvoi_x']
                        , 'dlt_x': row['dlt_x']
                        , 'actet_x': row['actet_x']
                        , 'clt_x': row['clt_x']
                        , 'latitude': row['latitude'] if len(row['latitude']) > 0 else '0.0'
                        , 'longitude': row['longitude'] if len(row['longitude']) > 0 else '0.0'
                        , 'my_size': num_candidate})
            
        #bi_query_names is defined in  __init__.py
        for query in bi_query_names:
            res = driver_elastic.request(query,elastic_index_bi, var)
            for i in range(len(res)):
                for item in res[i]:
                    # Le BI est forcement en base et donc la requete peut retourner systématique le bon siret
                    # a désactiver si la bdd s'alimente au fur et a mesure
                    # TODO ce test est completement artificiel ! mais:
                    # - il n'impacte pas la prod car normalement, le BI qu'on veut coder n'a pas de siretm, et on ne veut pas utiliser son siretc
                    # - il est nécessaire en training à moins de faire très attention à ne pas charger les données de test dans l'ES
                    if current_bi[i] != item['_id']:
                        echo_siret = get_bi_siret(item['_source'])  
                        if echo_siret:  # parfois le BI n'a pas encore de SIRET
                            dict_resp[current_bi[i]].append(Echo(siret=echo_siret, from_request = query, score_request = item["_score"]))
            
    return dict_resp
    

def get_most_similar_elastic(df, num_candidate = 10, 
                             async_driver_elastic=None, elastic_index_sirus='sirus_2020_e', chunksize = 50):
    """
    On execute les requetes ES pour récupérer les sirets les plus similaires.
    
    On sépare en deux lots : ceux qui ont été géocodé et les autres car sinon 
    les requêtes geoquery plantent.
    
    :params df: dataframe de bi
    :params num_candidate: nb de siret réponse par requete
    :param driver_elastic: driver elastic instancié. Si None: un sera créé avec les params par défaut
    :param elastic_index_sirus: index à interroger
    :param chunksize: nb de requetes à lancer dans un flow elastic
    :returns: dict_resp dict  pour clé cabbi et pour valeur liste de siret
    """
    dict_resp = {}
    list_cabbi = df.cabbi.values
    
    for cabbi in list_cabbi:
        dict_resp[cabbi] = []
        
    # remplacement des nan en vide
    df.replace({'latitude': {'nan': ''},
               'longitude': {'nan': ''}}, inplace=True)
    
    ########## NOUVEAU SYSTEME DE REQUETES ##########
 
    ##### REQUETES DE BASE, GEOCODAGE REUSSI OU NON #####
    ### NB : les requetes sont definies dans config/__init__.py

    if async_driver_elastic is None:
        async_driver_elastic = elastic.async_ElasticDriver(host=elastic_host, 
                                           port=elastic_port,
                                           requests_json_dir=os.path.realpath('elastic'))
 
    list_df = [df[i:i+chunksize] for i in range(0,df.shape[0],chunksize)]

    for part_df in list_df:
        current_bi = part_df.cabbi.values
        var = []
        
        for idx, row in part_df.iterrows():
            var.append({  'rs_x': row['rs_x']
                        , 'adr_et_voie_lib': row['geo_adresse']
                        , "depcom_g": row['clt_c_c']
                        , "clt_c_c": row['clt_c_c']
                        , "nomvoi_x": row['nomvoi_x']
                        , "actet_c_c": row['actet_c_c']
                        , "actet_x": row['actet_x']
                        , "profs_x": row['profs_x']
                        , "clt_x": row['clt_x']
                        , 'dlt_x': row['dlt_x']
                        , "latitude": row['latitude'] if len(row['latitude']) > 0 else '0.0'
                        , "longitude": row['longitude'] if len(row['longitude']) > 0 else '0.0'
                        , "my_size": num_candidate})

        res_queries=async_driver_elastic.loop.run_until_complete(async_driver_elastic.async_request(index=elastic_index_sirus,var=var))
        #async_driver_elastic.loop.close()
        for q,query in enumerate(res_queries):
            for b,bi in enumerate(query['responses']):
                for item in bi['hits']['hits']:
                    dict_resp[current_bi[b]].append(Echo(siret = item["_id"], from_request = query_names[q] , score_request = item["_score"]))
        
    ########## NOUVEAU SYSTEME DE REQUETE // FIN ##########
    
    return dict_resp


def get_naf_predictions_from_bdd(cabbis_to_fetch: list, 
                                 driver_bdd=None, naf_proj_bdd_table='naf_projections_2020'):
    """
    Récupération dans la BDD des prédictions NAF si elles existent

    :param cabbis_to_fetch: liste des cabbis
    :param driver_bdd: connexion à la base à utiliser. Si None: un sera créé avec les params par défaut
    :param naf_proj_bdd_table: table de la BDD à consulter
    :returns: dataframe calquée sur la structure de naf_proj_bdd_table avec le cabbi comme index
    """
    my_driver = driver_bdd if driver_bdd is not None else bdd.PostGre_SQL_DB(fs=fs, 
                                    fs_path_to_settings=minio_path_to_bdd_settings_file)
    
    sql_fetch_bi: str = f"SELECT * FROM {naf_proj_bdd_table} WHERE cabbi IN ("
    for cabbi in cabbis_to_fetch:
        sql_fetch_bi += f"'{cabbi}',"
    sql_fetch_bi = sql_fetch_bi[:-1] + ")"
    df_bi_naf = my_driver.read_from_sql(sql_fetch_bi)
    df_bi_naf = df_bi_naf.astype(str)
    df_bi_naf = df_bi_naf.set_index("cabbi")
    return df_bi_naf


def get_siret_ape_from_bdd(sirets_to_fetch: list, 
                           driver_bdd=None, 
                           sirus_bdd_table='siret_2019'):
    """
    Récupération des codes APE et APET dans la base

    :param sirets_to_fetch: liste des siret
    :param driver_bdd: connexion à la base à utiliser. Si None: un sera créé avec les params par défaut
    :param sirus_bdd_table: table de la BDD à consulter
    :returns: dataframe ['SIRET', 'APE', 'APET] avec le siret comme index
    """
    my_driver = driver_bdd if driver_bdd is not None else bdd.PostGre_SQL_DB(fs=fs, 
                                    fs_path_to_settings=minio_path_to_bdd_settings_file)
    
    sql_fetch_siret: str = f"SELECT CONCAT(sirus_id, nic) AS siret, ape, apet FROM {sirus_bdd_table} WHERE sirus_id || nic IN ("
    for siret in sirets_to_fetch:
        sql_fetch_siret += f"'{siret}',"
    sql_fetch_siret = sql_fetch_siret[:-1] + ")"
    df_siret = my_driver.read_from_sql(sql_fetch_siret)
    df_siret = df_siret.astype(str)
    df_siret = df_siret.set_index("siret")
    return df_siret


# def filtre_naf(siret_to_fetch: dict, predictions_naf: pd.DataFrame, 
#                driver_bdd=None, siret_bdd_table='siret_2019'):
#     """
#     Non utilisé.
#     Le but est de filtrer les sirets en ne gardant que ceux dont le code APE est dans les 
#     prédictions NAF (top10)
#     Au final on utilise le score naf autrement

#     :param sirets_to_fetch: dictionnaire {cabbi: list d'echos}
#     :param predictions_naf: DataFrame des prédictions contenant 'naf_code_i' pour i dans 0-9, indexée par le cabbi
#     :param driver_bdd: connexion à la base à utiliser
#     :param sirus_bdd_table: table de la BDD à consulter
#     :returns: dataframe ['SIRET', 'APE', 'APET] avec le siret comme index
#     """
#     my_driver = driver_bdd if driver_bdd is not None else bdd.PostGre_SQL_DB()

#     df_siret_naf = get_siret_ape_from_bdd(set_id_sirus, driver_bdd, sirus_bdd_table)
#     for cabbi in siret_to_fetch.keys():
#         naf_bis = predictions_naf.loc[cabbi][[f"naf_code_{i}" for i in range(10)]].values
#         indices_to_keep = []
#         for i, echo in enumerate(siret_to_fetch[cabbi]):
#             if len(echo.siret) > 0:
#                 naf_siret = df_siret_naf.loc[echo.siret]["apet"]
#                 if naf_siret in naf_bis:
#                     indices_to_keep.append(i)
#         siret_to_fetch[cabbi] = [siret_to_fetch[cabbi][i] for i in indices_to_keep]
                
#     return siret_to_fetch


def geocode_new_bi(df, addok_url=None):
    """
    Géocodage de BI via appel aux API insee (voir geocodage.py)

    le dossier contenant les fichiers de géocodage est "./geocodage"

    :params df: dataframe à coder
    :param addok_url: dictionnaire d'adresse des api ban, bano, poi
    :return: la dataframe enrichie des champs ['longitude', 'latitude', 'geo_score', 'geo_type', 'geo_adresse']
    """
    # TODO
    dir_path = os.path.dirname(os.path.realpath(__file__))
    keep_path = os.path.join(dir_path, "geocodage")
    data_path = os.path.join(keep_path, "data.csv")
    file_to_keep = ["geo_cog_new_2018.csv", "geo_cog_old_2018.csv" ]
    delete_all_files_execpt_from_directory(keep_path, file_to_keep)
    df.to_csv(data_path, index=False,sep=',')
    df_enriched = geo(data_path, addok_url, path_to_geodata=keep_path)
    dtypes = df_enriched.dtypes
    for col in df_enriched.columns:
        # cast to str for merge
        df_enriched[col] = df_enriched[col].apply(str)
    df_merged = df.merge(df_enriched, how='left', left_on='cabbi', right_on='cabbi',validate='one_to_one')
    for col in df_enriched.columns:
        # cast back
        if dtypes[col] != "object":
            df_merged[col] = df_merged[col].astype(dtypes[col])
    return df_merged
    

def generate_echos(input_df,
                   driver_bdd, bdd_naf_proj_table, naf_projection_func,
                   elastic_driver, async_elastic_driver, elastic_bi_index, elastic_sirus_index,
                   addok_urls=None):
    """
    Fonction permettant de générer des échos en récupérant les augmentations de 
    données disponibles

    :param input_df: dataframe à traiter
    :param driver_bdd: driver de requètes postgreSQL
    :param bdd_naf_proj_table: nom de la table dans laquelle les résultats du modèle NAF sur les BIs sont disponible
                               peut être None (et on ne va pas requeter la base)
    :param naf_projection_func: fonction permettant d'appliquer le modèle de projection NAF
                                    entrée : sous-ensemble de input_df
                                    sortie : df avec 'cabbi' en index et les colonnes 'naf_code_i' et 'naf_score_i' pour i dans [0, 9]
                                peut être None (et on n'appelera pas de modèle)
    :param elastic_driver: driver elastic search
    :param elastic_bi_index: nom de l'index contenant les données BI
    :param elastic_sirus_index: nom de l'index contenant les données Siret
    :param addok_urls: dictionnaire d'urls pour les services externes de géolocalisation:
                        { 'ban': ,'bano': , 'poi': ...}
                       pour géocoder les BI dont on n'a pas trouvé le géocodage dans ES
                       peut être None
    :return dict_echo: 
        - input_df augmentée des données géocodage et NAF
        - dictionnaire avec clé: cabbi, valeur: liste de siret (Echos), généré par get_most_similar_elastic
    """
    ############
    # enrichissement géocodage
    ############
    logging.info('Enrichissement BI... ')
    # récupération des enrichissements sur ES
    input_df = data_enrichment(input_df, 
                               driver_elastic=elastic_driver, 
                               elastic_index_bi=elastic_bi_index)
    
    if "_source_geo_flag" in input_df.columns:
        input_df = input_df.rename(columns={"_source_latitude":"latitude",
                                            "_source_longitude":"longitude",
                                            "_source_geo_adresse":"geo_adresse",
                                            "_source_geo_flag":"geo_flag"})
        
    # géocodage si non trouvé dans ES
    if addok_urls is not None:   
        cabbis_for_geoapi = input_df[~(input_df['geo_flag']==True)] if 'geo_flag' in input_df.columns else input_df
        if len(cabbis_for_geoapi):
            # call geo api
            logging.info("Appel de l'api de geo-encodage")
            for col in ['latitude', 'longitude', 'geo_adresse']:
                if col in cabbis_for_geoapi.columns:
                    cabbis_for_geoapi = cabbis_for_geoapi.drop([col], axis=1)
            cabbis_for_geoapi = geocode_new_bi(cabbis_for_geoapi, addok_urls)
            input_df = input_df.set_index('cabbi')
            for col in ['latitude', 'longitude', 'geo_adresse']:
                input_df.loc[cabbis_for_geoapi.cabbi, col] = cabbis_for_geoapi[col].apply(str).values
            input_df = input_df.reset_index()
        input_df["latitude"] = input_df["latitude"].astype(str)
        input_df["longitude"] = input_df["longitude"].astype(str)
    logging.info("Enrichissement terminé")

    ############
    #  Enrichissement NAF
    ############

    # Récupération des projections NAF ("vrais BIs", non modifiés par rapport à la BDD)
    input_df['actet_c_c'] = ""
    projs_naf_df = []
    if bdd_naf_proj_table is not None:
        logging.info('Récupération des prédictions NAF')
        proj_nafs = get_naf_predictions_from_bdd(input_df['cabbi'].values.tolist(), 
                                                driver_bdd=driver_bdd, naf_proj_bdd_table=bdd_naf_proj_table)
        input_df['actet_c_c'] = input_df.join(proj_nafs, 
                                              on='cabbi')['naf_code_0'].values
        projs_naf_df.append(proj_nafs)

    if naf_projection_func is not None:
        # Calcul des prédictions NAF non trouvées (les "faux BIs", requetes manuelles RECAP)
        # Elles ont actet_c_c = NaN. On va donc prédire le NAF pour elles
        cabbis_for_naf_model = input_df[input_df.actet_c_c.isna()]
        if len(cabbis_for_naf_model):
            logging.info('Calcul des prédictions NAF')
            proj_nafs2 = naf_projection_func(cabbis_for_naf_model)
            projs_naf_df.append(proj_nafs2)
        
    if len(projs_naf_df):
        projs_naf_df = pd.concat(projs_naf_df)

        projs_naf_df = projs_naf_df.reset_index()
        proj_cols = projs_naf_df.columns.difference(input_df.columns).tolist()
        for col in ['cabbi'] + proj_cols:
            projs_naf_df[col] = projs_naf_df[col].apply(str)
        input_df = input_df.merge(projs_naf_df[['cabbi'] + proj_cols], how='left')
        input_df['actet_c_c']= input_df['naf_code_0']
        input_df = input_df.fillna(0)
    
    
    ###############
    # Verification format
    ###############
    structure = ['geo_adresse', 
                    'clt_c_c',
                    'nomvoi_x',
                    'actet_c_c', 
                    'actet_x', 
                    'profs_x',
                    'clt_x', 
                    'latitude', 
                    'longitude']
    
    for col in structure:
        if col not in input_df:
            input_df[col] = ""


    #############
    # Recherche des échos soit dans SIRUS, soit dans les BI déjà traités (avec un SIRET attribué)
    #############
    logging.info('Requetes ES sirus...')
    sirus_to_fetch = get_most_similar_elastic(input_df, 
                                              async_driver_elastic=async_elastic_driver, 
                                              elastic_index_sirus=elastic_sirus_index)
    logging.info('Requetes ES sirus terminé')

    logging.info('Requetes ES BI...')
    # TODO: gérer plusieurs index ?
    similar_bi = get_most_similar_elastic_bi(input_df, 
                                             driver_elastic=elastic_driver,
                                             elastic_index_bi=elastic_bi_index)
    logging.info('Requetes ES BI terminé')
    for key in sirus_to_fetch.keys():
        sirus_to_fetch[key] = sirus_to_fetch[key] + similar_bi[key]

    return input_df, sirus_to_fetch


#######################
#   Results calculation
#######################

def get_top_k(dict_echo: dict, 
              bi_directory_path: str = "emb_bi", 
              nb_k: int = 10,
              naf_predictions_df=None,
              coeff_naf_score: float = 0,
              coeff_nb_fois_trouve: float = 0,
              driver_bdd=None,
              sirus_bdd_table='siret_2019',
              sirus_proj_bdd_table='sirus_projections_2020',
              naf_proj_bdd_table='naf_projections_2020'):
    """
    
    Récupère les top-k pour un BI donnée.
    
    Chargement de tous les siret et projections sirus associées.
    
    Calcul de la similarité cosinus entre un BI et la liste de siret.
    
    Trie par similarité (décroissant).
    
    :params dict_echo: dictionnaire ayant pour clé cabbi et pour valeur liste de siret, générée par get_most_similar_elastic
    :params bi_directory_path: dossier des embedding rp
    :params nb_k: taille du top-k
    :param naf_predictions_df: df contenant les prédictions NAF
    :param coeff_naf_score: coeff metamodel associé au score naf
    :param coeff_nb_fois_trouve: coeff metamodel associé au nb d'occurence ES
    :param driver_bdd: driver de requêtes postgreSQL
    :param sirus_bdd_table: table contenant les données sirus
    :param sirus_proj_bdd_table: table contenant les projections sirus
    :param naf_proj_bdd_table: table contenant les projections naf
    :returns: dict_topk dict pour clé cabbi et pour value list de siret triés par similarité cosinus
    """
    logger = logging.getLogger('runtime')
    dict_topk: dict = {}
    my_driver = driver_bdd if driver_bdd is not None else bdd.PostGre_SQL_DB(fs=fs, 
                                    fs_path_to_settings=minio_path_to_bdd_settings_file)


    ######
    # Score d'occurences ES
    ######
    nb_occ_of_siret_in_es_by_bi = {}  # nb occ in ES by cabbi then siret
    for cabbi in dict_echo.keys():
        nb_occ_of_siret_in_es_by_bi[cabbi] = Counter([echo.siret for echo in dict_echo[cabbi]])
    set_id_sirus = list(set(sum([list(echos_counter.keys()) for echos_counter in nb_occ_of_siret_in_es_by_bi.values()], [])))  # all sirets
    
    ######
    # Score naf
    ######

    # Récupération des projections NAF
    df_naf = None
    if naf_predictions_df is None:
        if naf_proj_bdd_table is not None:
            logger.info("Chargement projections naf...")
            cabbi_naf_to_fetch = dict_echo.keys()
            df_naf = get_naf_predictions_from_bdd(cabbi_naf_to_fetch, my_driver, naf_proj_bdd_table)
    else:
        df_naf = naf_predictions_df.copy(deep=True)
    if df_naf is not None:
        df_naf = pd.wide_to_long(df_naf, stubnames=["naf_code",'naf_score'], i='cabbi', j = "dummy",sep="_" ).reset_index().drop("dummy",axis=1).set_index(["cabbi","naf_code"])
    
    # Récupération des codes ape des entreprises
    logger.info("Chargement details siret...")
    df_siret_ape = get_siret_ape_from_bdd(set_id_sirus, driver_bdd=my_driver, sirus_bdd_table=sirus_bdd_table)
    
        
    ######
    #  Score SIRET
    ######
    # Récupération des projections siret
    sql_fectch = f"SELECT *, CONCAT(sirus_id, nic) AS siret FROM {sirus_proj_bdd_table} WHERE sirus_id || nic IN {tuple(set_id_sirus)} "
    my_driver = driver_bdd if driver_bdd is not None else bdd.PostGre_SQL_DB(fs=fs, 
                                    fs_path_to_settings=minio_path_to_bdd_settings_file)
    df_emb_sirus = my_driver.read_from_sql(sql_fectch).set_index("siret")
    
    # Chargement des projections bi (stockées en local pour l'instant)
    with open(os.path.join(bi_directory_path, "matrix_embedding.p"), "rb") as f:
        my_matrix = pickle.load(f)
        
    with open(os.path.join(bi_directory_path, "ids.p"), "rb") as f:
        cabbi = pickle.load(f)

    # Récupération des candidats et calcul des scores
    logger.info("Calcul des scores...")
    for idx, bi in enumerate(cabbi.itertuples()):
        vector_bi = my_matrix[idx]
        id_bi = bi.cabbi
        if id_bi in nb_occ_of_siret_in_es_by_bi:
            candidates = [] # liste des candidats siret avec leur projection
            for item_sirus in nb_occ_of_siret_in_es_by_bi[id_bi]:
                sirus_id = item_sirus[0:9]
                nic = item_sirus[-5:]
                siret_ = sirus_id + nic
                # On cherche le vector sirus
                if len(siret_) > 0:
                    try:
                        resp = df_emb_sirus.loc[siret_]
                    except KeyError:
                        resp = ''
                        logger.warning(f"siret {siret_} not found in projection database")
                else:
                    resp = ''
                # une projection sirus peut ne pas exister
                if len(resp) > 0:
                    candidates.append(resp)

            list_similarity = []
            for item in candidates:
                item_siret = item['sirus_id'] + item['nic']
                vector_si = item['embdedding']
                cosine_similarity = 1 - distance.cosine(vector_bi, vector_si)
                
                nb_fois_trouve = nb_occ_of_siret_in_es_by_bi[id_bi][item_siret]
                
                naf_score = 0
                if df_naf is not None and id_bi in df_naf.index:
                    if item_siret in df_siret_ape.index:
                        candidate_ape = df_siret_ape.loc[item_siret]['apet']
                        if candidate_ape in list(df_naf.loc[id_bi].index):
                            naf_score = float(df_naf.loc[id_bi].loc[candidate_ape]["naf_score"])

                final_similarity = cosine_similarity + coeff_nb_fois_trouve * nb_fois_trouve + coeff_naf_score * naf_score
                    
                list_similarity.append({  "cabbi": id_bi
                                        , "siret": item_siret
                                        , "similarite_final": final_similarity
                                        , "similarite": cosine_similarity
                                        , "nb_fois_trouve": nb_fois_trouve
                                        , "naf_score": naf_score
                                        , "codage_auto": 0
                                        , "codage_auto_proba": 0
                                       })
                
                
            list_similarity = sorted(list_similarity, key=lambda k: k['similarite_final'], reverse=True)
            dict_topk[id_bi] = list_similarity
        else:
            logger.warning("desynchro bi projection / bi topk")
            continue
            
            
    for key in dict_topk.keys():
        dict_topk[str(key)] = dict_topk[key][:nb_k]
            
    return dict_topk
            

#######################
#   Metrics
#######################
def elastic_recall(dict_recall: dict, df_groundtruth):
    """
    
    Calcule le pourcentage de siret retrouvé dans les requetes ES
    
    :param dict_recall:  dict pour clé cabbi et pour valeur liste de siret
    :param df_groundtruth: df contenant la ground truth
    :return siret_is_in_echo: dict pour clé cabbi, pour valeur 0 ou 1 si le siret-target a été trouvé dans les réponses
    """
    logger = logging.getLogger('runtime')    
    count_bi = len(dict_recall.keys())
    
    siret_is_in_echo = {}
    for key in dict_recall.keys():
        siret_is_in_echo[key] = 0
    
    for key in dict_recall.keys():
        if len(df_groundtruth.loc[key]) > 0:
            solution_sirus = df_groundtruth.loc[key]['sirus_id']
            solution_nic = df_groundtruth.loc[key]['nic']
            solution_id_sirus = solution_sirus + solution_nic
                            
            for echo in dict_recall[key]:
                if solution_id_sirus in echo.siret:
                    siret_is_in_echo[key] += 1
                    break
                
    count_is_in_echo = 0            
    for key in siret_is_in_echo.keys():
        count_is_in_echo += siret_is_in_echo[key]
    logger.info(f"siret_trouvé {count_is_in_echo} sur {count_bi} soit {(count_is_in_echo / count_bi) * 100} %")
        
    return siret_is_in_echo


###################
#   Export
###################
def export_topk_to_bdd(work_dir: str, topk: dict, len_topk: int,
                       sqltable_to_fill, driver_bdd=None):
    """
    Export des topk + embedding dans une bdd
    
    :param work_dir: dossier contenant les fichiers d'embeddings (ids.p, matrix_embedding.p)
    :param topk: dict contenant les topk à exporter
    :param len_topk: longueur du topk
    :param sqltable_to_fill: table à remplir
    :param driver_bdd: driver de bdd à utiliser
    """
    logger = logging.getLogger('runtime')
    
    with open(os.path.join(work_dir, "ids.p"), "rb") as f:
        cabbi = pickle.load(f)
    list_file_si = glob.glob(os.path.join(work_dir, "*matrix*.p"))
    list_file_si = sorted(list_file_si, key=str.lower)

    for i, file_si in enumerate(list_file_si):
        if i == 0:
            with open(file_si, "rb") as input_file_si:
                matrix_embedding_bi = pickle.load(input_file_si)
            full_bi = matrix_embedding_bi
        else:
            with open(file_si, "rb") as input_file_si:
                matrix_embedding_bi = pickle.load(input_file_si)
            matrix_embedding_bi = matrix_embedding_bi
            full_bi = np.vstack([full_bi, matrix_embedding_bi])
    
    df_bi_emb = pd.DataFrame()
    df_bi_emb['cabbi'] = cabbi['cabbi']
    df_bi_emb['embdedding'] = full_bi.tolist()
    
    
    # Formatage du dataframe pour obtnir la structure d'une table:
    # cabbi top_1_siret top_1_similarité top_2_siret top_2_similarité ...
    
    dict_flatten = {}
    
    for key in topk.keys():
        dict_flatten[key] = {}
        for i, candidat in enumerate(topk[key]):
            for nest_key in topk[key][i].keys(): 
                if nest_key not in ["cabbi"]:
                    dict_flatten[key][f"top_{i+1}_{nest_key}"] = topk[key][i][nest_key]
    
    df_topk = pd.DataFrame.from_dict(dict_flatten, orient='index')
    df_topk.index = df_topk.index.set_names(['foo'])
    df_topk = df_topk.reset_index().rename(columns={df_topk.index.name:'cabbi'})
    
    # print(df_topk.head())
    df_bi_topk = df_bi_emb.merge(df_topk, how='left', left_on='cabbi', right_on='cabbi', validate='one_to_one')
    # print(df_bi_topk.head())
    # print(df_bi_topk.info())
    my_driver = driver_bdd if driver_bdd is not None else bdd.PostGre_SQL_DB(fs=fs, 
                                    fs_path_to_settings=minio_path_to_bdd_settings_file)
    df_bi_topk.to_sql(sqltable_to_fill,
                      my_driver.engine,
                      if_exists = 'append',
                      index = False,
                      dtype = { 'embdedding' : sqlalchemy.dialects.postgresql.ARRAY(sqlalchemy.types.REAL)})    
    

###############################
#  Hacks (unused)
###############################

def reorder_topk_by_same_depcode(input_df, topk,
                                 bdd_sirus_table, driver_bdd=None):
    """
    Pour chaque BI, réordonner les échos en plaçant en premier les échos dont le code département matche avec
    le BI (on garde globalemnt l'ordre des 2 listes, match et non-match)

    pour les entreprises, on récupère l'adr_depcom dans la BDD
    pour le cabbi, on récupère dans input_df le dlt_x, ou le depcom_code si le dlt_x est vide.

    NON UTILISE : finalement le réordonnancement ne se fait qu'à l'affichage dans recap

    :param input_df: dataframe, données BI
    :param topk: résultat des fonctions précédentes (liste de résultat / cabbi)
    :param bdd_sirus_table: nom de la table sirus dans la BDD
    :param driver_bdd: driver BDD
    :returns: les topk avec l'ordre des résultats modifié (mais même structure)
    """
    my_driver = driver_bdd if driver_bdd is not None else bdd.PostGre_SQL_DB(fs=fs, 
                                    fs_path_to_settings=minio_path_to_bdd_settings_file)

    # récupération des informations de code département des entreprises
    all_sirets = set(sum([[k['siret'] for k in echos] for (_, echos) in topk.items()],[]))
    all_sirets_str = "('" + "','".join(all_sirets) + "')"
    req_depcom = f"SELECT CONCAT(sirus_id, nic) as siret, adr_depcom FROM {bdd_sirus_table} WHERE (sirus_id || nic) IN {all_sirets_str}"
    sirets_depcoms = my_driver.read_from_sql(req_depcom)

    def get_dep(siret):
        siret_depcom = sirets_depcoms[sirets_depcoms.siret == siret].adr_depcom.values.tolist()[0]
        return siret_depcom[:-3]

    # comparaison et réordonnancement
    for cabbi, results in topk.items():
        cabbi_series = input_df[input_df.cabbi == cabbi].iloc[0]
        cabbi_dlt_x = cabbi_series.dlt_x if cabbi_series.dlt_x else cabbi_series.clt_c_c[:-3]
        
        same_dep_sirets = [res['siret'] for res in results if get_dep(res["siret"]) == cabbi_dlt_x]
        same_dep_res = []
        other_dep_res = []
        for res in results:
            if res["siret"] in same_dep_sirets:
                same_dep_res.append(res)
            else:
                other_dep_res.append(res)
        topk[cabbi] = same_dep_res + other_dep_res

    return topk
            

def main():
    """
    
    Script qui simule l'arrivé d'un BI à prédire.
    
    Calcul des métrique, et stockage en local ou bdd.
    
    CREATE VIEW view_test_2019 AS SELECT r.* 
    FROM rp_final_2019 r 
    WHERE r.cabbi NOT IN (SELECT cabbi FROM cabbi_test) 
    
    """
    with open(os.path.join(os.path.dirname(__file__), 'logging.conf.yaml'), 'r') as stream:
        log_config = yaml.load(stream, Loader=yaml.FullLoader)
    logging.config.dictConfig(log_config)
    logger = logging.getLogger('runtime')
    logger.info("Commencement script_runtime")

    ###############
    #   Variables
    ###############
    model_directory = 'trainings/2021-02-18_25'
    conf_file = os.path.join(model_directory, 'config_bi.yaml')

    work_dir = os.path.join(model_directory, 'runtime')
    BI_EMB_DIRECTORY = os.path.join(work_dir, 'emb_bi')

    # Data
    minio_endpoint = 'http://minio.ouest.innovation.insee.eu'
    minio_path_to_bdd_settings_file = 's3://ssplab/aiee2/data/settings.yml'
    rp_table = "rp_2020"
    sirus_table = "sirus_2020"
    sirus_proj_table = "sirus_projections_2020"
    naf_proj_table = "naf_projections_2020"

    addok_apis = {
        'ban': 'http://api-ban.ouest.innovation.insee.eu/search',
        'bano': 'http://api-bano.ouest.innovation.insee.eu/search',
        'poi': 'http://api-poi.ouest.innovation.insee.eu/search'
    }

    elastic_host = ["http://es01-elastic-ssplab.marathon-ouest.containerip.dcos.thisdcos.directory:9200","http://es02-elastic-ssplab.marathon-ouest.containerip.dcos.thisdcos.directory:9200","http://es03-elastic-ssplab.marathon-ouest.containerip.dcos.thisdcos.directory:9200"]
    elastic_port = 9200
    elastic_index_bi = 'rp_2020_e'
    elastic_index_sirus = 'sirus_2020_e'

    # données pour training metamodèle
    # gt_pickle_file = os.path.join(model_directory, 'df_solution_eval.p')
    # données pour eval finale
    gt_pickle_file = os.path.join(model_directory, 'df_solution_test.p')
    with open(gt_pickle_file, "rb") as input_file:
        df_groundtruth = pickle.load(input_file)
        df_groundtruth = df_groundtruth.set_index("cabbi")
    cabbis_list = "'" + "','".join(df_groundtruth.index.to_list()) + "'"
    sql_bi = f"SELECT r.* FROM {rp_table} r INNER JOIN {sirus_table} s ON (LEFT(r.siretc,9) = s.sirus_id) AND s.nic = RIGHT(r.siretc, 5) WHERE r.cabbi IN ({cabbis_list}) AND siretc IS NOT NULL LIMIT 100000;"

    export_to_bdd = True
    result_db_table = 'bi_topk_test_refacto_final'

    compute_metrics = True
    results_file = os.path.join(work_dir, 'runtime_metrics_final.json')

    LEN_TOPK: int = 10

    #################
    #   Run
    #################
    # chargement modèle. MetaModel is only loaded if exists
    configs, cleaners, processes, model, meta_model, threshold = load_pipeline_and_model(conf_file, for_training=True)
    
    # setup environment
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(BI_EMB_DIRECTORY, exist_ok=True)

    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': minio_endpoint})
    driver_bdd = bdd.PostGre_SQL_DB(fs=fs, 
                                    fs_path_to_settings=minio_path_to_bdd_settings_file)
    elastic_driver = elastic.ElasticDriver(host=elastic_host, 
                                           port=elastic_port,
                                           requests_json_dir=os.path.realpath('elastic'))
    async_elastic_driver = elastic.async_ElasticDriver(host=elastic_host, 
                                           port=elastic_port,
                                           requests_json_dir=os.path.realpath('elastic'))

    ###################
    #   Process
    ###################
    logger.info("Chargement BI...")
    siret_in_es_echos = {}
    siret_ranks = {}
    for df in driver_bdd.read_from_sql_with_chunksize(sql_bi, chunksize=1000):
        # clean directory
        shutil.rmtree(BI_EMB_DIRECTORY)
        os.makedirs(BI_EMB_DIRECTORY)
        # project BIs
        logger.info('Projection BI...')
        project(df, 
                configs, cleaners, processes, model,
                projections_dir=BI_EMB_DIRECTORY, work_dir=os.path.join(work_dir, 'tmp'))
        logger.info("Projection terminée")

        # get echos (no NAF)
        df, echos = generate_echos(df,
                                   driver_bdd, naf_proj_table, None,
                                   elastic_driver, async_elastic_driver, elastic_index_bi, elastic_index_sirus,
                                   addok_urls=addok_apis)

        # get top-k
        logger.info('topk...')
        topk = get_top_k(echos,
                         bi_directory_path=BI_EMB_DIRECTORY, 
                         nb_k=LEN_TOPK,
                         naf_predictions_df=df,
                         coeff_naf_score=configs['final_score_coeffs']['coeff_naf_score'],
                         coeff_nb_fois_trouve=configs['final_score_coeffs']['coeff_nb_fois_trouve'],
                         driver_bdd=driver_bdd,
                         sirus_bdd_table=sirus_table,
                         sirus_proj_bdd_table=sirus_proj_table,
                         naf_proj_bdd_table=naf_proj_table)
        logger.info('topk terminé')

        if meta_model is not None:
            logger.info('codage automatique...')
            topk = codage_automatique(topk, meta_model, threshold)
            logger.info('codage automatique terminé')

        # calcul perfs
        if compute_metrics:
            # TODO : adapt and merge results on all datafram
            df_siret_in_es_echos = elastic_recall(echos, df_groundtruth)
            siret_in_es_echos.update(df_siret_in_es_echos)

            df_siret_rank = metric_top(topk, df_groundtruth)
            siret_ranks.update(df_siret_rank)
        
        # export
        if export_to_bdd:
            logger.info('Export...')
            export_topk_to_bdd(BI_EMB_DIRECTORY, topk, LEN_TOPK, result_db_table)
            logger.info('Export terminé')

    if compute_metrics:
        # TODO : save global results
        final_metrics = {
            'nb_cabbis': len(siret_in_es_echos),
            'es_recall': sum(siret_in_es_echos.values()) / len(siret_in_es_echos),
            'topk_pct': {
                "1": len([v for v in siret_ranks.values() if v == "1"]),
                "3": len([v for v in siret_ranks.values() if v == "3"]),
                "5": len([v for v in siret_ranks.values() if v == "5"]),
                "10": len([v for v in siret_ranks.values() if v == "10"]),
            }
        }
        with open(results_file, 'w') as f:
            json.dump(final_metrics, f)
        ext = results_file.split('.')[-1]
        with open(results_file.replace(ext, 'siret_in_echos.'+ext), 'w') as f:
            json.dump(siret_in_es_echos, f)
        with open(results_file.replace(ext, 'siret_ranks.'+ext), 'w') as f:
            json.dump(siret_ranks, f)


    
if __name__ == '__main__':
    main()
    
    
