#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# 
# 
#
# @author Brivaël Sanchez
# @date: 2020/21/12

import sys
sys.path.append('..')
import os
import yaml
import s3fs
from datetime import date
import glob
import logging
import pickle

import pandas as pd
from numpy import load

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


def load_pipeline_and_model(conf_file: str, fs=None, for_runtime=True):
    """
    Charge les classes du pipeline utilisé par le modèle
    
    :params conf_file: nom du fichier de configuration
    :param fs: s3fs filesystem (si none, un s3fs sera instancié avec les parametres par défaut)
    :param for_runtime: mettre à True si runtime (on enlève les colonnes de GT de la config du cleaner, car pas présentes dans les prédictions runtime)
    :return configs, cleaners, processes, model: classe du preprocessing du model et le model
    """
    # Init meta variable
    today = date.today().isoformat()
    current_dir = os.path.dirname(__file__)
    path_run = None
    id_run = None

    logger = None
    thismodule = sys.modules[__name__]
    
    if conf_file is None:
        conf_file = os.path.join(current_dir, 'config.yaml')
    with open(conf_file) as f:
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
                if for_runtime:
                    for col in ['siretm', 'siretc']:
                        if col in cleaner[obj]['cols']:
                            cleaner[obj]['cols'].pop(cleaner[obj]['cols'].index(col))
                instance = class_(cleaner[obj]['cols'])
                cleaners.append(instance)   
    except ValueError:
        logger.error(F"Erreur dans la creation des cleaners")
        sys.exit(2)
    try:
        processes = []
        for transformer in configs['processes']:
            for obj in transformer.keys():
                class_ = getattr(thismodule, transformer[obj]['type'])
                attr = transformer[obj]['param']
                attr['path_run'] = path_run
                attr['id_run'] = id_run
                instance = class_(**attr)
                processes.append(instance)
    except ValueError:
        logger.error(F"Erreur dans la creation des Processes")
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
                
    model = list_model[0] # TODO:  On a forcément qu'un seul model
    
    return configs, cleaners, processes, model


def project(data, configs, cleaners, processes, model, 
            work_dir=None, model_dir=None):
    """
    Projecte un jeu de donnée dans un espace de représentation.
    Nécéssite un model de similarité déjà entrainée

    IMPORTANT: ne fonctionne a priori que pour le ModelTransformer...

    :param data: dataframe pandas à processer (format BI)
    :param configs: config chargée via load_pipeline_and_model
    :param cleaners: cleaners chargés via load_pipeline_and_model
    :param processes: processes chargés via load_pipeline_and_model
    :param model: modèle chargé via load_pipeline_and_model, avec les poids chargés via model.load_model
    :param work_dir: directory ou enregistrer les fichiers de runtime
    :param model_dir: directory où trouver les fichiers modèles (chargés par les process)
    :returns: np.array[bool, 1] : True si non codable, np.array[float, 2]: scores des classes (non-codable: classe 1)
    """
    ###################
    # Overwrite / clean tmp file
    ###################
    try:
        mydir = work_dir if work_dir is not None else os.path.dirname(os.path.realpath(__file__))
        tmp_dir = os.path.join(mydir, "tmp")
        os.makedirs(tmp_dir, exist_ok=True)
        buffer_dir = os.path.join(mydir, "buffer")
        os.makedirs(buffer_dir, exist_ok=True)
        output_dir = os.path.join(mydir, "output")
        os.makedirs(output_dir, exist_ok=True)
        model_dir = model_dir if model_dir is not None else os.path.join(mydir, "input")

        filelist = glob.glob(os.path.join(mydir, "*.csv"))
        for f in filelist:
            os.remove(f)
        filelist = glob.glob(os.path.join(mydir, "*.npz"))
        for f in filelist:
            os.remove(f)
        filelist = glob.glob(os.path.join(mydir, "*.npy"))
        for f in filelist:
            os.remove(f)

        filelist = [f for f in os.listdir(tmp_dir)]
        for f in filelist:
            os.remove(os.path.join(tmp_dir, f))  

        filelist = [f for f in os.listdir(buffer_dir)]
        for f in filelist:
            os.remove(os.path.join(buffer_dir, f))

    except ValueError:
        print(F"Erreur dans la suppresion des fichier temporaire")
        logging.getLogger().error(F"Erreur dans la suppresion des fichier temporaire")
        sys.exit(2)

    print("Writing data...")
    data_file = os.path.join(mydir, "dataset.csv")
    data.to_csv(data_file, sep=';',index=False)

    print("get id")
    ids = pd.read_csv(data_file, sep=';', dtype='str')
    with open(os.path.join(output_dir, "ids.p"), "wb") as output_file:
        pickle.dump(ids, output_file)

    try:
        for cleaner in cleaners:
            list_of_files = glob.glob(f'{tmp_dir}/*') 
            # Initialization
            if len(list_of_files) == 0:
                latest_file = os.path.join(mydir, "dataset.csv")
            else:
                latest_file = max(list_of_files, key = os.path.getctime)
            cleaner.process(latest_file, tmp_dir)
    except ValueError:
        print(F"Erreur dans l'execution de {cleaner.__class__.__name__}")
        logging.getLogger().error(F"Erreur dans l'execution de {cleaner.__class__.__name__}")
        sys.exit(2)

    for process in processes:
        list_of_files = glob.glob(f'{tmp_dir}/*') 
        latest_file = max(list_of_files, key = os.path.getctime)
        process.run_model(latest_file, 
                          os.path.join(model_dir, "tokenizer.p"),
                          os.path.join(model_dir, "dict_info.p"),
                          buffer_dir)

    list_of_encoded_data_files = glob.glob(f'{buffer_dir}/*')     

    for file in list_of_encoded_data_files:
        
        X = load(file, allow_pickle=True)
        predictions = model.predict(X)
        if 'decision_threshold' in configs:
            return predictions[:, 1] >= configs['decision_threshold'], predictions
        return predictions[:, 1] >= 0.5, predictions
        
#    if export:
#        move_all_files_from_directory_to_another(SOURCE_DIR, EMB_DIRECTORY)
#        export_sirus_to_bdd(EMB_DIRECTORY)


if __name__ == '__main__':

    model_dir = 'model_output'
    work_dir = 'test_runtime'
    os.makedirs(work_dir, exist_ok=True)

    model_config, model_cleaners, model_processes, model = \
                    load_pipeline_and_model(os.path.join(model_dir, "config.yaml"))
    model.load_model(os.path.join(model_dir, 'model.h5'),
                        os.path.join(model_dir, 'dict_info.p'))

    test_data = [
        {
            "cabbi": 'test_cabbi',
            "rs_x": "COLLEGE GEORGES MANDEL",
            "clt_x": "SOULAC SUR MER",
            "profs_x": "ASSISTANTE D EDUCATION",
            "profi_x": "",
            "profa_x": "",
            "numvoi_x": "",
            "typevoi_x": "",
            "actet_x": "ACCUEIL DES COLLEGIENS EDUCATION",
            "dlt_x": "33",
            "plt_x": "",
            "vardompart_x": "" 
        },
        {
            "cabbi": 'test_cabbi2',
            "rs_x": "COLLEGE GEORGES MANDEL",
            "clt_x": "SOULAC SUR MER",
            "profs_x": "ASSISTANTE D EDUCATION",
            "profi_x": "",
            "profa_x": "",
            "numvoi_x": "",
            "typevoi_x": "",
            "actet_x": "ACCUEIL DES COLLEGIENS EDUCATION",
            "dlt_x": "33",
            "plt_x": "",
            "vardompart_x": "" 
        }
    ]

    input_df = pd.DataFrame(test_data)
    predictions = project(input_df, model_config, model_cleaners, model_processes, model, 
                          work_dir=work_dir, model_dir=model_dir)
    print('Predictions: ')
    print(predictions) 