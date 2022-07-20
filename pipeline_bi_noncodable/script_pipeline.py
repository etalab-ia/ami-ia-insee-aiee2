"""
Fichier qui charge des fichier de configuration puis le excute.

Les variables suivantes doivent être dans l'environnement: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN

Gère le flux de processing et d'entrainement.
Gère la sauvegarde et le nettoyage des fichiers de résultats intermédiare et des modèles
Author : bsanchez@starclay.fr
date : 06/08/2020
"""

import sys
sys.path.append("..")
import os
import os.path
from os import path
import pandas as pd
import numpy as np
import shutil
import glob
import yaml
import json
import getopt
import pickle
from datetime import date
import botocore

import logging
import logging.config
with open(os.path.join(os.path.dirname(__file__), 'logging.conf.yaml'), 'r') as stream:
    log_config = yaml.load(stream, Loader=yaml.FullLoader)
logging.config.dictConfig(log_config)

import data_import.bdd as bdd
import training_classes.preprocessing as pre

from training_classes.CleanerRemoveSpecialChar import CleanerRemoveSpecialChar

from training_classes.ProcessBigram import ProcessBigram
from training_classes.ProcessTfIdf import ProcessTfIdf
from training_classes.ProcessDoc2Vec import ProcessDoc2Vec
from training_classes.ProcessEmbedding import ProcessEmbedding
from training_classes.ProcessFastText import ProcessFastText

from training_classes.ModelTree import Tree
from training_classes.ModelXGB import XGBoost
from training_classes.ModelLogisticRegression import LogRegr
from training_classes.ModelLogisticRegressionCV import LogRegrCV
from training_classes.ModelNaiveBayes import NaiveBaye
from training_classes.ModelSMOTE import Smote
from training_classes.ModelSvm import Svm
from training_classes.ModelMLP import MLP
from training_classes.ModelMLPFusion import MLPFusion 
from training_classes.ModelTransformer import MLPTransformer
from training_classes.ProcessMLP import ProcessMLP


from training_classes.utils import *

from sklearn.model_selection import train_test_split

import scipy.sparse
from scipy.sparse import coo_matrix, hstack

import s3fs
from numpy import load


def help():
    print('**** PIPELINE *****')
    print("Le pipeline clean, transforme puis fais applique les modèles contenu dans le fichier de config")
    print("Les variables suivantes doivent être dans l'environnement: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN")
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

logger = logging.getLogger('Pipeline')

conf_file = None

based = False
skip = False

full_cmd_arguments = sys.argv
argument_list = full_cmd_arguments[1:]
short_options = "c:hbs"
long_options = ["config", "help", "based", "skip"]

try:
    arguments, values = getopt.getopt(argument_list, short_options, long_options)
except getopt.error as err:
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
    logger.error(F"Erreur dans le chargement du fichier de configuration {conf_file}")
    sys.exit(2)
            
####################
# Running information
# Generation de l'id (unique) du run 
####################

today = date.today().isoformat()

# Les variables de connexion doivent être dans l'environnement: AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN
fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': configs['minio']['endpoint']})

not_exist = True

BUCKET = "s3://ssplab/aiee2/non_codable_new/"

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

save_file_on_minio("config.yaml", dir_name=path_run)
os.makedirs("output", exist_ok=True)
shutil.copy("config.yaml", "output/config.yaml")

if len(arguments) == 0:
    arguments = 'no args'

with open(os.path.join(current_dir, "output", "args.txt"),"w") as f:
    f.write(str(arguments))
save_file_on_minio("output/args.txt", dir_name=path_run)


########
# INIT
# Instantiation des opération contenues dans le fichier de config avec leurs 
# paramètres
########

thismodule = sys.modules[__name__]

try:
    cleaners = []
    for cleaner in configs['cleaners']:
        for obj in cleaner.keys():
            class_ = getattr(thismodule, cleaner[obj]['type'])
            instance = class_(cleaner[obj]['cols'])
            cleaners.append(instance)   
except ValueError:
            logger.error(F"Erreur dans la creation des cleaners")
            sys.exit(2)

try:
    Processes = []
    for process in configs['processes']:
        for obj in process.keys():
            class_ = getattr(thismodule, process[obj]['type'])
            attr = process[obj]['param']
            attr['path_run'] = path_run
            attr['id_run'] = id_run
            instance = class_(**attr)
            Processes.append(instance)
except ValueError:
            logger.error(F"Erreur dans la creation des Processes")
            sys.exit(2)
            
try:
    list_echantillon = []
    for model in configs['echantillon']:
        for obj in model.keys():
            class_ = getattr(thismodule, model[obj]['type'])
            attr = model[obj]['param']
            attr['path_run'] = path_run
            attr['id_run'] = id_run
            instance = class_(**attr)
            list_echantillon.append(instance)
except ValueError:
            logger.error(F"Erreur dans la creation des echantilloneurs")
            sys.exit(2)
            
try:
    list_model = []
    for model in configs['models']:
        for obj in model.keys():
            class_ = getattr(thismodule, model[obj]['type'])
            attr = model[obj]['param']
            attr['path_run'] = path_run
            attr['id_run'] = id_run
            instance = class_(**attr)
            list_model.append(instance)
except ValueError:
            logger.error(F"Erreur dans la creation des modeles")
            sys.exit(2)

###################
# Overwrite / clean tmp file
###################
try:
    mydir = os.path.dirname(os.path.realpath(__file__))

    filelist = glob.glob(os.path.join(mydir, "*.csv"))
    for f in filelist:
        os.remove(f)

    filelist = glob.glob(os.path.join(mydir, "*.npz"))
    for f in filelist:
        os.remove(f)
        
    filelist = glob.glob(os.path.join(mydir, "*.npy"))
    for f in filelist:
        os.remove(f)

    if based:
        for file in fs.ls(F"{BUCKET}based",refresh=True):
            fs.rm(file)
        fs.touch(F"{BUCKET}based/.keep")

    os.makedirs("tmp", exist_ok=True)
    filelist = [f for f in os.listdir("tmp/")]

    for f in filelist:
        os.remove(os.path.join("tmp/", f))  
except ValueError:
    logger.error(F"Erreur dans la suppresion des fichier temporaire")
    sys.exit(2)

####################
# import data
# Requete SQL dans la base postgreSQL
####################

if skip is False:
    try:
        my_driver = bdd.PostGre_SQL_DB(logger=logger)
        sql = configs['import_data_sql']
        for df in my_driver.read_from_sql(sql, chunksize=60000):
            with open("dataset.csv", 'a') as f:
                    df.to_csv(f, mode='a', sep=';', header=f.tell()==0,index=False)    
    except ValueError:
            logger.error(F"Erreur dans l'importation des données")
            sys.exit(2)

####################
# Process: Cleaner
####################

    # test dataset 
#     df_year = pd.read_csv("dataset.csv", usecols=['year'], sep=';')
    

    try:
        for cleaner in cleaners:
            cleaner.process("dataset.csv")
    except ValueError:
            logger.error(F"Erreur dans l'execution de {cleaner.__class__.__name__}")
            sys.exit(2)

    # save target
    try:
        df_target = pd.read_csv("tmp/1.csv",usecols=['target'],sep=';')
        sp_target = scipy.sparse.csr_matrix(df_target.target.values)
        sp_target = sp_target.T
        scipy.sparse.save_npz("sp_target.npz",sp_target)
        del df_target
        del sp_target
        save_file_on_minio("sp_target.npz", dir_name=path_run)
    except ValueError:
        logger.error(F"Erreur dans la sauvegarde des target")
        sys.exit(2)
    ###############################
    # processes
    ###############################
    

    for Processe in Processes:
        list_of_files = glob.glob('tmp/*') 
        latest_file = max(list_of_files, key = os.path.getctime)
        Processe.apply(latest_file)
        del Processe


    try:
        list_of_files = glob.glob('tmp/*') 
        latest_file = max(list_of_files, key=os.path.getctime)
        logger.info(f'latest transformed data file: {latest_file}')
        if latest_file.endswith('.npz'):
            shutil.copyfile(latest_file, "dataset_transformed.npz") 
            save_file_on_minio("dataset_transformed.npz",path_run)
        if latest_file.endswith('.csv'):
            shutil.copyfile(latest_file, "dataset_transformed.csv") 
            save_file_on_minio("dataset_transformed.csv",path_run)
        if latest_file.endswith('.npy'):
            shutil.copyfile(latest_file, "dataset_transformed.npy") 
            save_file_on_minio("dataset_transformed.npy",path_run)
            
        if path.exists("target.npy"):   #généré par ProcessMLP
            save_file_on_minio("target.npy",path_run)
        if path.exists("embedding_fasttext.npy"):   #généré par ProcessMLP
            save_file_on_minio("embedding_fasttext.npy",path_run)
    except ValueError:
        logger.error(F"Erreur dans la récupération du dataset transformer")
        sys.exit(2)
    except (botocore.exceptions.ClientError, IOError):
        logger.error(f'error uploading the dataset to minio')

    ###############################
    # train_test_split
    # Appel de méthode de sur-échantillonage
    ###############################
#     latest_file = "dataset_transformed.npy"
    if latest_file.endswith('.npz'):
        X = scipy.sparse.load_npz('dataset_transformed.npz')
    if latest_file.endswith('.csv'):
        X = pd.read_csv("dataset_transformed.csv",sep=';')
    if latest_file.endswith('.npy'):
        X = load("dataset_transformed.npy",allow_pickle=True)
        
    if path.exists("target.npy"):
        labels = np.load("target.npy")
    else:
        labels = scipy.sparse.load_npz('sp_target.npz').toarray()
        
    cabbis = pd.read_csv('dataset.csv', sep=';')['cabbi'].values.tolist()

else:
    load_file_on_minio(F"{BUCKET}based/dataset.csv","dataset.csv")
    load_file_on_minio(F"{BUCKET}based/dataset_transformed.npy","dataset_transformed.npy")
    load_file_on_minio(F"{BUCKET}based/embedding_fasttext.npy","embedding_fasttext.npy")
    # load_file_on_minio(F"{BUCKET}based/sp_target.npz","sp_target.npz")
    load_file_on_minio(F"{BUCKET}based/target.npy","target.npy")
    #labels = scipy.sparse.load_npz('sp_target.npz').toarray()
    X = load("dataset_transformed.npy")
    labels = np.load("target.npy")
    cabbis = pd.read_csv('dataset.csv', sep=';')['cabbi'].values.tolist()

### train test split classique
    
range_train, range_test, y_train, y_test = train_test_split(list(range(len(labels))), labels, test_size=0.3, stratify=labels)

cabbis_train = np.asarray(cabbis)[range_train].tolist()
X_train = X[range_train]

cabbis_test = np.asarray(cabbis)[range_test].tolist()
X_test = X[range_test]

with open('cabbis_train.json', 'w') as f:
    json.dump(cabbis_train, f)
    
with open('cabbis_test.json', 'w') as f:
    json.dump(cabbis_test, f)
    
save_file_on_minio("cabbis_train.json",path_run)
save_file_on_minio("cabbis_test.json",path_run)

if based is False:
    # ravel()
    y_train = y_train
    y_test = y_test

    for ech in list_echantillon:
        X_train, y_train = ech.apply(X_train,y_train)

    ###############################
    # Model
    ###############################
    for model in list_model:
        model.apply(X_train, X_test, y_train, y_test)


for f in os.listdir("."):
    if f[:8] == "metrics_" and f[-4:] == ".csv":
        shutil.move(f, "output/metrics.csv")
        
for f in os.listdir("output"):
    if not os.path.isdir(os.path.join("output", f)):
        save_file_on_minio(os.path.join("output", f), path_run)