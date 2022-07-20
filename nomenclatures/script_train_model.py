#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# train_model.py
#
# script de training du modèle:
# - génération des données préprocessées, ou chargement si elles existent
# - séparation en train/validation/test
# - train
# - calcule des données post-process nécessaires pour faire des prédictions à partit du modèle entrainé
#
# Les données sont sauvegardées dans des dossiers communs à tous les trainings
# le training est sauvegardé dans un nouveau dossier
#
# Le training utilise config.yaml, qui est copié dans le nouveau dossier
#
# @author cyril.poulet@starclay.fr
# @date: oct 2020

import tensorflow as tf
tf.executing_eagerly()
# tf.compat.v1.disable_eager_execution() # MEMORY LEAK OTHERWISE
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import numpy as np
import pandas as pd
import ast
import logging
import logging.config
import os
import shutil
import yaml
import json
from datetime import date, datetime
import s3fs

from sklearn.model_selection import train_test_split
import fasttext.util

import sys
sys.path.append('..')
from data_import.bdd import PostGre_SQL_DB
from training_classes.cleaner import Cleaner
from training_classes.nomenclature import Nomenclature
from training_classes.fasttext_singleton import FasttextSingleton
from training_classes.embedding_dictionary import EmbeddingDictionary
from training_classes.nomenclature_distance import NomenclatureDistance
from training_classes.anchor_positive_pairs_batch import AnchorPositivePairsBatch
from training_classes.similarity_model import SimilarityModel
from training_classes.training_model import TrainingModel
from training_classes.lstm_model import LstmModel
from training_classes.transformer_model import TransformerModel
from training_utils import load_config, save_config, push_to_minio, get_trainings_dir, get_last_local_training_dir

import argparse

"""
Script de training

"""

####################
# Logging
# Chargement du fichier de configuration log
####################
with open(os.path.join(os.path.dirname(__file__), 'logging.conf.yaml'), 'r') as stream:
    log_config = yaml.load(stream, Loader=yaml.FullLoader)
logging.config.dictConfig(log_config)

logger = logging.getLogger()

####################
# Parametres
# Chargement du fichier de configuration
####################

parser = argparse.ArgumentParser(
    description="Script pour entraîner un modèle de type similarité à une nomenclature")
parser.add_argument("config_file", nargs='?', type=str, default='config.yaml',
                    help="fichier de config")
args = parser.parse_args()

if hasattr(args, 'config_file') and  args.config_file:
    conf_file = os.path.abspath(args.config_file)
else:
    current_dir = os.path.dirname(__file__)
    conf_file = os.path.join(current_dir, 'config.yaml')


try:
    print(F"Chargement du fichier de configuration {conf_file}")
    logger.info(F"Chargement du fichier de configuration {conf_file}")
    with open(conf_file) as f:
        configs = yaml.safe_load(f)
except ValueError:
    print(F"Erreur dans le chargement du fichier de configuration {conf_file}")
    logger.error(F"Erreur dans le chargement du fichier de configuration {conf_file}")
    sys.exit(2)


####################
# Running information
# Generation de l'id (unique) du run 
####################
            
today = date.today().isoformat()

fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': configs['minio']['endpoint']})
sync_with_minio = configs['minio']['sync']

NOMENCLATURE_NAME = configs['data']['nomenclature']['name']
BASE_REMOTE_DIR = configs['minio']['trainings_dir']
TRAININGS_LOCAL_DIR, TRAININGS_REMOTE_DIR = get_trainings_dir(configs)

# global cleaning
# shutil.rmtree(os.path.join('trainings', NOMENCLATURE_NAME), ignore_errors=True)
# fs.rm(TRAININGS_REMOTE_DIR, recursive=True)
# exit()
not_exist = True
try:
    if sync_with_minio:
        for file in fs.ls(TRAININGS_REMOTE_DIR,refresh=True):
            if os.path.basename(file) == today:
                not_exist = False
                list_directory = fs.ls(os.path.join(TRAININGS_REMOTE_DIR, today),refresh=True)
                list_directory = [os.path.basename(file) for file in list_directory]
                list_directory = [int(x)for x in list_directory]
                id_run = max(list_directory) + 1
                fs.touch(os.path.join(TRAININGS_REMOTE_DIR, today, str(id_run), "0"))
                break

        if not_exist:
            id_run = 1
            fs.touch(os.path.join(TRAININGS_REMOTE_DIR, today, "0"),create_parents=True)
            fs.touch(os.path.join(TRAININGS_REMOTE_DIR, today, str(id_run), "0"),create_parents=True)
        remote_path_run = os.path.join(TRAININGS_REMOTE_DIR, today, str(id_run))
        id_run = f"{today}_{id_run}"
    else:
        # purely local
        remote_path_run = ''
        os.makedirs(TRAININGS_LOCAL_DIR, exist_ok=True)
        last_training = os.path.basename(get_last_local_training_dir(TRAININGS_LOCAL_DIR))
        lt_date, lt_ind = last_training.split('_')
        lt_date = datetime.strptime(lt_date, "%Y-%m-%d").date()
        lt_ind = int(lt_ind)
        if lt_date == date.today():
            id_run = lt_ind + 1
        else:
            id_run = 0
        id_run = f"{today}_{id_run}"
except ValueError:
    print(F"Erreur dans la creation de l'id_run")
    logger.error(F"Erreur dans la creation de l'id_run")
    sys.exit(2)


local_path_run = os.path.join(TRAININGS_LOCAL_DIR, id_run)
if not os.path.exists(local_path_run):
    os.makedirs(local_path_run)

with open(os.path.join(local_path_run, "paths.json"), 'w') as f:
    json.dump( {'remote_path': remote_path_run}, f)

# add log in training dir
training_log_file = os.path.join(local_path_run, 'training.log')
formatter = logging.Formatter(log_config['formatters']['simple']['format'])
ch = logging.FileHandler(training_log_file)
ch.setLevel('DEBUG')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger = logging.getLogger('NomenclatureTraining')

save_config(configs, local_path_run)
if sync_with_minio:
    push_to_minio(local_path_run)

####################
# Preprocess data
#
# 1. on calcule les différents emplacements data
# 2. on regarde si des data préprocessées correspondant aux paramêtres du training existent en local, puis en remote
# 3. Si ce n'est pas le cas, on préprocess les données, on les sauve et on les push sur minio
####################

local_data_dirname = 'data'

if 'use_stemmer' in configs['data'] and configs['data']['use_stemmer']:
    local_data_dirname += '_stemmed'

use_ngrams = False
if 'ngrams' in configs['data'] and configs['data']['ngrams']['use']:
    use_ngrams = True
    local_data_dirname += f"_{configs['data']['ngrams']['ngrams_size']}grams"

use_fasttext = False
if 'fasttext' in configs['data'] and configs['data']['fasttext']['use']:
    use_fasttext = True
    local_data_dirname += f"_fasttext_{configs['data']['fasttext']['embedding_size']}"

remote_path_data = os.path.join(TRAININGS_REMOTE_DIR, local_data_dirname)
local_data_path = os.path.join(TRAININGS_LOCAL_DIR, local_data_dirname)

if not os.path.exists(local_data_path):
    os.makedirs(local_data_path)

bdd = PostGre_SQL_DB()


def check_local_data():
    """
    Vérification que les données préprocessées existent en local ou en remote.
    Si c'est le cas, au besoin on récupère dpuis le remote, puis on tente de charger les données
    et on vérifie qu'elles correspondent aux paramêtres de training
    """
    logger.info('Checking local data')

    cleaner_save_file = os.path.join(local_data_path, configs['data']['nomenclature']['topcode'])
    data_cleaner = None
    ngrams_size = None
    if use_ngrams:
        ngrams_size = configs['data']['ngrams']['ngrams_size']
    try:
        cleaner_class = Cleaner.load_factory(cleaner_save_file)
    except ValueError:
        try:
            cleaner_class = Cleaner.create_factory(configs['data']['cleaner_class'])
        except ValueError as e:
            logger.error(e)
            return False
    try:
        data_cleaner = cleaner_class(bdd, cleaner_save_file, 
                                    use_stemmer=configs['data']['use_stemmer'],
                                    use_fasttext=use_fasttext,
                                    ngrams_size=ngrams_size)
    except ValueError as e:
        pass

    if not os.path.exists(cleaner_class.get_sqlfile_path(cleaner_save_file)):
        # get data on minio
        logger.info('... getting data from minio')
        if sync_with_minio:
            for f in fs.ls(remote_path_data):
                if os.path.basename(cleaner_class.get_rawdata_path(cleaner_save_file)) in f:
                    continue
                fs.get(f, os.path.join(local_data_path, os.path.basename(f)))

    logger.info('... trying to load from disk')
    try:
        naf_distances = NomenclatureDistance.load(bdd, cleaner_save_file + "_nomdistance")
    except:
        logger.info(f'... no data on disk')
        return False
        
    try:
        sql_request = configs['data']['postgres_sql']
        if 'limit_nb_doc' in configs['data'] and configs['data']['limit_nb_doc'] is not None:
            sql_request += f" limit {configs['data']['limit_nb_doc']}"
        if data_cleaner is None:
            data_cleaner = cleaner_class(bdd, cleaner_save_file, 
                                        use_stemmer=configs['data']['use_stemmer'],
                                        use_fasttext=use_fasttext,
                                        ngrams_size=ngrams_size)
        data_cleaner.get_and_clean_data(sql_request, naf_distances.nomenclature)
    except ValueError as e:
        logger.error(f'SQL request in config does not correspond to the data files found on disk ({cleaner_save_file}). Please clean before relaunching')
        raise e

    if not os.path.exists(data_cleaner.prepared_data_file):
        return False

    return True


def generate_data():
    """
    Génération des données préprocessées :
    - récupération dans la base
    - clean
    - calcul du vocabulaire
    - projection des données dans le vocabulaire
    - préparation de la nomenclature
    - sauvegarde en local et sur minio
    - si on utilise fasttext, on s'assure que le modèle est disponible en local
    """
    logger.info('Generating data')
    # Fetch naf data
    logger.info('... getting nomenclature')
    naf = Nomenclature(bdd, 
                       configs['data']['nomenclature']['topcode'],
                       configs['data']['nomenclature']['node_dist_top_to_first_cat'])

    # Get BI data
    logger.info('... getting data from db')
    sql_request = configs['data']['postgres_sql']
    if 'limit_nb_doc' in configs['data'] and configs['data']['limit_nb_doc'] is not None:
        sql_request += f" limit {configs['data']['limit_nb_doc']}"

    cleaner_save_file = os.path.join(local_data_path, configs['data']['nomenclature']['topcode'])

    ft = None
    if use_fasttext:
        ft = FasttextSingleton(local_path='fasttext', 
                               remote_endpoint=configs['minio']['endpoint'],
                               remote_path=os.path.join(BASE_REMOTE_DIR, configs['data']['fasttext']['remote_directory']),
                               embeddings_dim=configs['data']['fasttext']['embedding_size'])
        ft.get_model_files()
    ngrams_size = None
    if use_ngrams:
        ngrams_size = configs['data']['ngrams']['ngrams_size']
    
    cleaner_class = Cleaner.create_factory(configs['data']['cleaner_class'])
    data_cleaner = cleaner_class(bdd, cleaner_save_file, 
                                 use_stemmer=configs['data']['use_stemmer'],
                                 use_fasttext=use_fasttext,
                                 ngrams_size=ngrams_size)
    data_cleaner.get_and_clean_data(sql_request, naf)

    logger.info('... creating vocabulary')
    embeddings_voc_path = data_cleaner.create_dictionary(naf)

    logger.info('... preparing data')
    data_cleaner.put_training_data_in_voc()

    logger.info('... preparing nomenclature')
    embeddings_voc = EmbeddingDictionary.load(embeddings_voc_path)
    naf.build_nomenclature_embeddings(embeddings_voc)

    # build target distance
    logger.info('... calculating classes distances')
    naf_distances = NomenclatureDistance(naf)
    
    # save nomenclature from nom_distance
    naf_distances.save(cleaner_save_file + "_nomdistance")

    # freeing memory
    if ft is not None:
        ft.delete_model()

    #### copy all on minio
    if sync_with_minio:
        logger.info('... saving to minio')
        fs.put(local_data_path, remote_path_data, recursive=True)

# on cherche en local, et si besoin on génère
try:
    if not check_local_data() or configs['data']['force_generation']:
        generate_data()
except Exception as e:
    logger.error(f'Error preparing data: {e}. Please clean and restart')
    exit(-1)


#######################
# prepared data is locally available, let's load it
# Available columns : cabbi;actet_c;actet_c_libelle;actet_repr;rs_repr;prof_repr
######################

logger.info('Loading data')
cleaner_save_file = os.path.join(local_data_path, configs['data']['nomenclature']['topcode'])
data_cleaner = Cleaner.load_factory(cleaner_save_file)(bdd, cleaner_save_file)

generic = lambda x: ast.literal_eval(x)
conv = {'actet_repr': generic,
        'rs_repr': generic,
        'prof_repr': generic}
prepared_data = pd.read_csv(data_cleaner.prepared_data_file, sep=';', converters=conv)
naf_distances = NomenclatureDistance.load(bdd, cleaner_save_file + "_nomdistance")
naf = naf_distances.nomenclature
embeddings_voc = naf.embeddings_dict
logger.info('... data loaded')

###########
# Create training data : split train/validation/test
# Available columns : cabbi;actet_c;actet_c_libelle;actet_repr;rs_repr;prof_repr
###########
logger.info('Separating in training/eval/test')
training_conf = configs['trainings']

Y = prepared_data[[training_conf['data']['gt_column'], 'cabbi']]
X = prepared_data[training_conf['data']['input_columns']]

seq_len_X = sum([max(X[col].apply(len)) for col in training_conf['data']['input_columns']])
seq_len_nom = max([len(naf.get_nomenclature_embeddings(n)) for n in naf.nodes])
seq_len = max(seq_len_nom, seq_len_X) + 1
training_conf['model_params']['seq_len'] = seq_len
save_config(configs, local_path_run)
if sync_with_minio:
    push_to_minio(local_path_run)

X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.3, random_state=42)

X_train, X_valid, y_train, y_valid = train_test_split(X_train , y_train, test_size=0.2, random_state=42)

y_test[['cabbi']].to_csv(os.path.join(local_path_run, "cabbi_test.csv"), sep=';', index=False)
if sync_with_minio:
    push_to_minio(local_path_run)
    
########
# Define model
########
logger.info('Defining model')
model_config = training_conf['model_params']
model_config['nb_fields'] = len(training_conf['data']['input_columns']) + 1
model_config['vocab_size'] = embeddings_voc.nb_different_tokens
if use_fasttext:
    model_config['pre_trained_embeddings_weights'] = embeddings_voc.get_embeddings()
    model_config['pre_trained_weights_trainable'] =  configs['data']['fasttext']['trainable']
    model_config['embedding_size'] = configs['data']['fasttext']['embedding_size']
model = eval(training_conf['model'])(local_path_run, naf_distances, load_path=None, **model_config)


##########
# Training
#########

logger.info('Defining training')

batch_size = training_conf['training_params']['batch_size']
nb_batches_train = int(len(X_train) / batch_size)
nb_batches_valid = int(len(X_valid) / batch_size)
nb_batches_test = int(len(X_test) / batch_size)
nb_epochs = training_conf['training_params']['nb_epochs']

logger.info('Training')

train_data = AnchorPositivePairsBatch(nomenclature_distance=naf_distances,
                                      seq_len=seq_len)
train_data.set_data(input_classes=y_train[[training_conf['data']['gt_column']]].to_numpy(),
                    input_fields=X_train.to_numpy(),
                    num_batchs=nb_batches_train,
                    batch_size=batch_size)

valid_data = AnchorPositivePairsBatch(nomenclature_distance=naf_distances,
                                      seq_len=seq_len)
valid_data.set_data(input_classes=y_valid[[training_conf['data']['gt_column']]].to_numpy(),
                    input_fields=X_valid.to_numpy(),
                    num_batchs=nb_batches_valid,
                    batch_size=batch_size)
    
test_data = AnchorPositivePairsBatch(nomenclature_distance=naf_distances,
                                     seq_len=seq_len)
test_data.set_data(input_classes=y_test[[training_conf['data']['gt_column']]].to_numpy(),
                   input_fields=X_test.to_numpy(),
                   num_batchs=nb_batches_test,
                   batch_size=batch_size)
test_data.save(os.path.join(local_path_run, "batcher"))
if sync_with_minio:
    push_to_minio(local_path_run)

history = model.train_model(train_data, valid_data, nb_epochs=nb_epochs)
logger.info(f'... training history : {history.history["loss"]}')
pd.DataFrame(history.history).to_csv(os.path.join(local_path_run, "training_history.json"), sep=';', index=False)

##############
# post-training
#############

logger.info('building nomenclature projections')
naf.build_nomenclature_projections(lambda x : model.run_model_single_side([np.array([emb], dtype=np.float32) for emb in test_data.format_input(x)])[0])
naf.create_trigram_repr()
test_data.save(os.path.join(local_path_run, "batcher"))
if sync_with_minio:
    push_to_minio(local_path_run)
