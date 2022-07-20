#!/usr/bin/env python3
"""
Fichier qui charge des fichier de configuration puis le excute.
Gère le flux de processing et d'entrainement.
Gère la sauvegarde et le nettoyage des fichiers de résultats intermédiare et des modèles
Author : bsanchez@starclay.fr
date : 06/08/2020
"""

import sys
sys.path.append("..")
import os

import numpy as np
import shutil
import glob
import yaml
import json
import getopt
import pickle
from datetime import date
import random
import s3fs
import logging
import logging.config

import pandas as pd
import numpy as np
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import scipy.sparse

import data_import.bdd as bdd
import training_classes.preprocessing as pre

from training_classes.CleanerRemoveSpecialChar import CleanerRemoveSpecialChar
from training_classes.CleanerAdrCltX import CleanerAdrCltX
from training_classes.CleanerAdrDltX import CleanerAdrDltX
from training_classes.CleanerConcat_profs import CleanerConcat_profs
from training_classes.ProcessMLPSiamese import ProcessMLPSiamese
from training_classes.ModelSimilarity import MLPSimilarity
from training_classes.utils import *

from config import *

def help():
    print('**** PIPELINE *****')
    print("Le pipeline clean, transforme puis fais applique les modèles contenu dans le fichier de config")
    print("Voir Readme")
    print("Options supportées :")
    print("  -c   --config     préciser le fichier de config de l'opération (défaut: config.yaml)")
    print("  -s   --skip     skip la phase de cleaning/transform et utilise les fichiers du dossier based")
    print("  -b   --base     clean/transform dans le dossier based pour pouvoir les skipper à l'avenir. Supprime les anciens fichier dans le dossier based")
    print("  -h   --help       afficher l'aide")
    
####################
# Args
# Gestion des arguments 
####################

with open(os.path.join(os.path.dirname(__file__), 'logging.conf.yaml'), 'r') as stream:
    log_config = yaml.load(stream, Loader=yaml.FullLoader)
logging.config.dictConfig(log_config)

logger = logging.getLogger('Pipeline')

conf_file = None
based : bool = False
skip : bool = False
local_extracted_data_dir = 'trainings/data_210_fasttext' # used if skip = True. It is your responsibility to know if it is the correct data
# Le dossier doit contenir : dataset.csv, dataset_transformed.npy, dict_info.pickle, tokenizer.p, et si besoin embedding_fasttext.npy. Ces fichiers sont à récupérer d'un entrainement précédent avec la même config data

BUCKET : str = "s3://ssplab/aiee2/production_test/"
LOCAL_TRAINING_DIR = 'trainings'

full_cmd_arguments = sys.argv
argument_list = full_cmd_arguments[1:]
short_options = "c:hbs"
long_options = ["config", "help", "based", "skip"]

try:
    arguments, values = getopt.getopt(argument_list, short_options, long_options)
except getopt.error as err:
    print("Erreur dans la lecture des arguments")
    sys.exit(2)

for current_argument, current_value in arguments:
    if current_argument in ("-c", "--config"):
        if len(current_value):
            conf_file = current_value
    if current_argument in ("-b", "--based"):
            based = True
    if current_argument in ("-s", "--skip"):
            skip = True
    if current_argument in ("-h", "--help"):
        help()
        sys.exit(0)

####################
# Parametres
# Chargement du fichier de configuration
####################

current_dir = os.path.dirname(__file__)
if conf_file is None:
    conf_file = os.path.join(current_dir, 'config.yaml')

try:
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
#fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': configs['minio']['endpoint']})

not_exist : bool = True
try:
    if based is False:
        for file in fs.ls(BUCKET,refresh=True):
            if os.path.basename(file) == today:
                not_exist = False
                list_directory = fs.ls(f"{BUCKET}{today}/",refresh=True)
                list_directory = [os.path.basename(file) for file in list_directory]
                list_directory = [int(x)for x in list_directory]
                id_run = max(list_directory) + 1
                fs.touch(F"{BUCKET}{today}/{id_run}/0")
                break

        if not_exist:
            id_run = 1
            fs.touch(F"{BUCKET}{today}/0",create_parents=True)
            fs.touch(F"{BUCKET}{today}/{id_run}/0",create_parents=True)
        path_run = f"{BUCKET}{today}/{id_run}"
        id_run = f"{today}_{id_run}"
    else:
        path_run = f"{BUCKET}based"
        id_run = f"{today}_based"
except ValueError:
        logger.error(F"Erreur dans la creation de l'id_run")
        sys.exit(2)

local_path_training = os.path.join(LOCAL_TRAINING_DIR, id_run)
os.makedirs(local_path_training, exist_ok=True)
logger.info(f"Ce training sera sauvé sur minio à l'adresse {path_run}")
logger.info(f"Ce training sera sauvé localement dans {local_path_training}")

if len(arguments) == 0:
    arguments = 'no args'

shutil.copy(conf_file, os.path.join(local_path_training, 'config.yaml'))
with open(os.path.join(local_path_training, "paths.json"), 'w') as f:
    json.dump( {'remote_path': path_run}, f)
with open(os.path.join(local_path_training,"args.txt"),"w") as f:
    f.write(str(arguments))
push_to_minio(local_path_training, fs=fs)


########
# INIT
# Instantiation des opération contenues dans le fichier de config avec leurs 
# paramètres
########

thismodule = sys.modules[__name__]
logger.info('Instanciation de la pipeline')
try:
    cleaners = []
    for cleaner in configs['cleaners']:
        for obj in cleaner.keys():
            class_ = getattr(thismodule, cleaner[obj]['type'])
            instance = class_(cleaner[obj]['cols'])
            cleaners.append(instance)   
except ValueError as e:
    logger.error(F"Erreur dans la creation des cleaners : {str(e)}")
    sys.exit(2)

try:
    processes = []
    for transformer in configs['processes']:
        for obj in transformer.keys():
            class_ = getattr(thismodule, transformer[obj]['type'])
            attr = transformer[obj]['param']
            attr['local_path_run'] = local_path_training
            instance = class_(**attr)
            processes.append(instance)
except ValueError as e:
    logger.error(F"Erreur dans la creation des Processes : {str(e)}")
    sys.exit(2)
            
try:
    list_echantillon = []
    for model in configs['echantillon']:
        for obj in model.keys():
            class_ = getattr(thismodule, model[obj]['type'])
            attr = model[obj]['param']
            attr['local_path_run'] = local_path_training
            attr['id_run'] = id_run
            instance = class_(**attr)
            list_echantillon.append(instance)
except ValueError as e:
    logger.error(F"Erreur dans la creation des echantilloneurs : {str(e)}")
    sys.exit(2)
            
try:
    list_model = []
    for model in configs['models']:
        for obj in model.keys():
            class_ = getattr(thismodule, model[obj]['type'])
            attr = model[obj]['param']
            attr['local_path_run'] = local_path_training
            instance = class_(**attr)
            list_model.append(instance)
except ValueError as e:
    logger.error(F"Erreur dans la creation des modeles : {str(e)}")
    sys.exit(2)

tmp_dir = os.path.join(local_path_training, 'tmp')
os.makedirs(tmp_dir, exist_ok=True)
output_dir = os.path.join(local_path_training, 'output')
os.makedirs(output_dir)

###################
# Overwrite / clean tmp file
###################
try:
    if based:
        for file in fs.ls(F"{BUCKET}based",refresh=True):
            fs.rm(file)
        fs.touch(F"{BUCKET}based/.keep")
except ValueError:
    logger.error(F"Erreur dans la suppression des fichier temporaire")
    sys.exit(2)

####################
# import data
# Requete SQL dans la base postgreSQL
####################

if skip is False:
    logger.info('Récupération des données')
    dataset_file = os.path.join(local_path_training, "dataset.csv")
    try:
        my_driver = bdd.PostGre_SQL_DB(host=db_host,
                                       port=db_port,
                                       dbname=db_dbname,
                                       user=db_user,
                                       password=db_password,
                                       logger=logger)
        sql = configs['import_data_sql']
        for df in my_driver.read_from_sql(sql,chunksize=60000):
            with open(dataset_file, 'a') as f:
                df.to_csv(f, mode='a', sep=';', header=f.tell()==0,index=False)
    except ValueError:
        logger.error(F"Erreur dans l'importation des données")
        sys.exit(2)

####################
# Process: Cleaner
####################
    logger.info('Nettoyage des données')
    for cleaner in cleaners:
        list_of_files = glob.glob(f'{tmp_dir}/*') 
        # Initialization
        if len(list_of_files) == 0:
            input_file = dataset_file
            output_file = os.path.join(tmp_dir, "1.csv")
        else:
            input_file = max(list_of_files, key = os.path.getctime)
            output_file = os.path.join(tmp_dir, incrementTmpFile(input_file) + ".csv")
        cleaner.process(input_file, output_file)

    # save ID, cabbi/siret
    df_meta = pd.read_csv(dataset_file, usecols=['lot_id','siretc','cabbi'], sep=';', dtype=str)
    df_meta = df_meta.fillna('inconnue')
    annuaire = df_meta['lot_id'].values
    sirets = df_meta['siretc'].values
    cabbi = df_meta['cabbi'].values
    le = preprocessing.LabelEncoder()
    sirets = le.fit_transform(sirets)
    sirets = [[x] for x in sirets]

    ###############################
    # Processes
    ###############################
    logger.info('Préparation des données')
    for process in processes:
        list_of_files = glob.glob(f'{tmp_dir}/*') 
        input_file = max(list_of_files, key = os.path.getctime)
        output_file = os.path.join(tmp_dir, incrementTmpFile(input_file) + ".npy")
        if hasattr(process, 'train'):
            process.train(input_file)
        process.run(input_file, output_file)
        del process
    try:
        list_of_files = glob.glob(f'{tmp_dir}/*') 
        latest_file = max(list_of_files, key=os.path.getctime)
        ext = latest_file.split('.')[-1]
        dataset_transformed_file = os.path.join(local_path_training, f"dataset_transformed.{ext}")
        shutil.copyfile(latest_file, dataset_transformed_file)
        # shutil.rmtree(tmp_dir)         
    except ValueError:
        logger.error(F"Erreur dans la récupération du dataset transformer")
        sys.exit(2)

    ###############################
    # train_test_split
    # Appel de méthode de sur-échantillonage
    ###############################
    logger.info('Chargement des données de training')
    if dataset_transformed_file.endswith('.npz'):
        X = scipy.sparse.load_npz(dataset_transformed_file)
    if dataset_transformed_file.endswith('.csv'):
        X = pd.read_csv(dataset_transformed_file)
    if dataset_transformed_file.endswith('.npy'):
        X = np.load(dataset_transformed_file, allow_pickle=True)
    if dataset_transformed_file.endswith('.p'):
        X = pickle.load(open(dataset_transformed_file, "rb" ))

else:
    if local_extracted_data_dir is not None:
        logger.info(f'Récupération des données de training depuis {local_extracted_data_dir}')
        for f in os.listdir(local_extracted_data_dir):
            if os.path.isdir(os.path.join(local_extracted_data_dir, f)):
                continue
            shutil.copy(os.path.join(local_extracted_data_dir, f), 
                        os.path.join(local_path_training, f))
    else:
        #some files are missing here
        logger.info('Récupération des données de training depuis minio')
        load_file_on_minio(F"{BUCKET}based/dataset_transformed.npy",
                           os.path.join(local_path_training,"dataset_transformed.npy"),
                          fs=fs)
        load_file_on_minio(F"{BUCKET}based/sp_target.npz",
                           os.path.join(local_path_training,"sp_target.npz"),
                          fs=fs)
    logger.info('Chargement des données de training')
    # save ID, cabbi/siret
    dataset_file = os.path.join(local_path_training, "dataset.csv")
    df_meta = pd.read_csv(dataset_file, usecols=['lot_id','siretc','cabbi'], sep=';', dtype=str)
    df_meta = df_meta.fillna('inconnue')
    annuaire = df_meta['lot_id'].values
    sirets = df_meta['siretc'].values
    cabbi = df_meta['cabbi'].values
    le = preprocessing.LabelEncoder()
    sirets = le.fit_transform(sirets)
    sirets = [[x] for x in sirets]
    del df_meta
    # X = scipy.sparse.load_npz('dataset_transformed.npz')
    X = np.load(os.path.join(local_path_training,"dataset_transformed.npy"), allow_pickle=True)

#####
# TRAIN/TEST SPLIT
####
logger.info('Préparation du training')
logger.info(f'Le dataset contient {len(X[0])} examples')

# get lot_id or vague_id in order to sort
X_train = []
y_train = []
X_test = []
y_test = []
annuaire_test = []
annuaire_train_val = []
sirets_test = []
sirets_train_val = []
cabbi_test = []
cabbi_train_val = []

for row in range(len(X[0])):
    if "inconnue" in annuaire[row] :
        X_test.append(X[0][row])
        y_test.append(X[1][row])
        annuaire_test.append(annuaire[row])
        sirets_test.append(sirets[row])
        cabbi_test.append(cabbi[row])
    else:
        X_train.append(X[0][row])
        y_train.append(X[1][row])
        annuaire_train_val.append(annuaire[row])
        sirets_train_val.append(sirets[row])
        cabbi_train_val.append(cabbi[row])
            
# Matrix distance too large otherwise
if len(X_test) > 50000:
    index_elem_to_keep = random.sample(range(0, len(X_test) - 1), 50000)
    X_test = [X_test[i] for i in index_elem_to_keep]   
    y_test = [y_test[i] for i in index_elem_to_keep]
    annuaire_test = [annuaire_test[i] for i in index_elem_to_keep]
    sirets_test = [sirets_test[i] for i in index_elem_to_keep]
    cabbi_test = [cabbi_test[i] for i in index_elem_to_keep]
               

rs_v = ShuffleSplit(n_splits = 1, test_size=.33, random_state=42)
for train_index, test_index in rs_v.split(X_train):
        
    X_val = [X_train[i] for i in test_index] 
    y_val = [y_train[i] for i in test_index] 

    X_train = [X_train[i] for i in train_index] 
    y_train = [y_train[i] for i in train_index] 
        
    annuaire_val = [annuaire_train_val[i] for i in test_index]
    annuaire_train = [annuaire_train_val[i] for i in train_index]
    
    sirets_val = [sirets_train_val[i] for i in test_index]
    sirets_train = [sirets_train_val[i] for i in train_index]
    
    cabbi_val = [cabbi_train_val[i] for i in test_index]
    cabbi_train =  [cabbi_train_val[i] for i in train_index]
    
X_train = zip(X_train, y_train, annuaire_train, sirets_train, cabbi_train)
X_train = sorted(X_train, key = lambda tup: tup[2])
y_train = [x[1] for x in X_train]
sirets_train = [x[3] for x in X_train]
cabbi_train = [x[4] for x in X_train]
X_train = [x[0] for x in X_train]

X_test = zip(X_test, y_test, annuaire_test, sirets_test, cabbi_test)
X_test = sorted(X_test, key = lambda tup: tup[2])
y_test = [x[1] for x in X_test]
sirets_test = [x[3] for x in X_test]
cabbi_test = [x[4] for x in X_test]
X_test = [x[0] for x in X_test]

X_val = zip(X_val, y_val, annuaire_val, sirets_val, cabbi_val)
X_val = sorted(X_val, key = lambda tup: tup[2])
y_val = [x[1] for x in X_val]
sirets_val = [x[3] for x in X_val]
cabbi_val = [x[4] for x in X_val]
X_val = [x[0] for x in X_val]

with open(os.path.join(local_path_training, "sirets_train.p"), "wb") as f:
    pickle.dump(sirets_train, f)
    
with open(os.path.join(local_path_training, "sirets_test.p"), "wb") as f:
    pickle.dump(sirets_test, f)
    
with open(os.path.join(local_path_training, "sirets_val.p"), "wb") as f:
    pickle.dump(sirets_val, f)
    
with open(os.path.join(local_path_training, "cabbi_train.p"), "wb") as f:
    pickle.dump(cabbi_train, f)
    
with open(os.path.join(local_path_training, "cabbi_test.p"), "wb") as f:
    pickle.dump(cabbi_test, f)
    
with open(os.path.join(local_path_training, "cabbi_val.p"), "wb") as f:
    pickle.dump(cabbi_val, f)

logger.info(f'Le dataset de training contient {len(cabbi_train)} examples')

if based is False:
    if len(list_echantillon):
        logger.info('Echantillonnage des données')
        for ech in list_echantillon:
            X_train, y_train = ech.apply(X_train, y_train, sirets_train)

    ###############################
    # Model
    ###############################
    for i, model in enumerate(list_model):
        logger.info(f'Début trainings {i}')
        train_config = list(configs['models'][i].values())[0]['training']
        model.train_model(X_train, y_train, sirets_train, X_val, y_val, sirets_val, **train_config)
        model.save_model(os.path.join(local_path_training, f'model_{i}'))
        push_to_minio(local_path_training, fs=fs)