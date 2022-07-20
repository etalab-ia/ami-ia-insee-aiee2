#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# run_top_k.py training_dir weight_dir
#     training_dir : dossier du training. 
#                    Défault : training le plus récent dans config['local']['trainings_dir']/config['data']['nomenclature']['name']
#     weights_dir : nom du dossier de poids à utiliser dans le training/train_weights
#                   Défaut : le meilleur modèle dans training_dir
#
# script de calcul des performances du modèle:
# - fonction de chargement du modèle
# - fonction de prédiction d'un batch/d'un BI
# - fonction de calcul des topk pour un BI "raw" (non préprocessé)
# - fonction de calcul des performances topk d'un modèle sur les données de test sauvegardées au training
# - script permettant l'application de la dernière fonction à un modèle donné (par défaut: le dernier entrainé), de sauvegarder les résultats et de les uploader
#
# utilise config.yaml dans le dossier de training
#
# @author cyril.poulet@starclay.fr
# @date: oct 2020

import sys
sys.path.append('..')
import os
import yaml
import json
from tensorflow import keras
import numpy as np
import pandas as pd
import ast
import logging.config
import s3fs
import tempfile

from data_import.bdd import PostGre_SQL_DB
from training_classes.cleaner import Cleaner
from training_classes.nomenclature import Nomenclature
from training_classes.embedding_dictionary import EmbeddingDictionary
from training_classes.nomenclature_distance import NomenclatureDistance
from training_classes.anchor_positive_pairs_batch import AnchorPositivePairsBatch
from training_classes.similarity_model import SimilarityModel
from training_classes.training_model import TrainingModel
from training_classes.lstm_model import LstmModel
from training_classes.transformer_model import TransformerModel
from training_utils import load_config, push_to_minio, get_trainings_dir, get_last_local_training_dir, get_best_savedmodel
from script_train_top1_autovalidation_classifier import Top1ValidatorModel
import s3fs


def load_model_from_save_dir(model_save_dir, set_of_weights=None, bdd=None, fs=None):
    """
    Chargement de l'ensemble des objets nécessaire à prédire une projection

    :param model_save_dir: chemin de sauvegarde au training
    :param set_of_weights: nom du set de poids enregistrés à charger
    :returns: nomenclature : Nomenclature
              data_cleaner : NafCleaner
              data_formatter: AnchorPositivePairsBatch
              model: TrainingModel
              top1_model: model de classification (validation du top1) s'il existe, sinon None
    """
    config = load_config(model_save_dir)
    if bdd is None:
        bdd = PostGre_SQL_DB(host=db_host, port=db_port, dbname=db_dbname, user=db_user, password=db_password)
    cleaner_class = Cleaner.create_factory(config['data']['cleaner_class'])
    data_cleaner = cleaner_class(bdd, "")
    data_formatter = AnchorPositivePairsBatch.load(bdd, os.path.join(model_save_dir, "batcher"))
    nomenclature_distance = data_formatter.nomenclature_distance
    nomenclature = nomenclature_distance.nomenclature

    if set_of_weights is None:
        set_of_weights = get_best_savedmodel(model_save_dir)
    model = eval(config['trainings']['model'])(None, 
                                               nomenclature_distance, 
                                               load_path=os.path.join(model_save_dir, "train_weights", set_of_weights))

    top1_model = None
    try:
        top1_model = Top1ValidatorModel(model_save_dir)
    except (KeyError,FileNotFoundError, IsADirectoryError):
        pass
    return nomenclature, data_cleaner, data_formatter, model, top1_model

# final top-k score

def project_input(*args, data_formatter: AnchorPositivePairsBatch, model: TrainingModel):
    """
    Projette un input depuis les données cleanées des BIs (actet_x, rs_x, prof_x)

    :param args: listes des valeurs (liste simple pour un BI, liste de listes pour plusieurs)
    :param data_formatter: AnchorPositivePairsBatch utilisé lors du training
    :param model: TrainingModel entrainé
    :returns: projection si un seul BI, ou list de projections
    """
    single_mode = False
    if not isinstance(args[0], list):
        single_mode = True
        args = [[a] for a in args]
    batch_inputs = []
    for bi_fields in zip(*args):
        # convertir en indices, puis formatter / calculer les positions et champs
        bi_projections = [data_formatter.get_embeddings_voc().convert_voc(field) for field in bi_fields]
        batch_inputs.append(list(data_formatter.format_input(*bi_projections)))
        
    # réorganisation des dimensions en 3 vecteurs (tokens, champs, positions)
    batch_inputs = zip(*batch_inputs)
    batch_inputs = [np.array(batch_input, dtype=np.float32) for batch_input in batch_inputs]
    # projection
    text_projections_tf = model.run_model_single_side(batch_inputs)
    text_projections = text_projections_tf.numpy().tolist()
    if single_mode:
        return text_projections[0]
    return text_projections


def get_projection(input_df,
                   input_columns,
                   data_cleaner: Cleaner,
                   data_formatter: AnchorPositivePairsBatch,
                   model: TrainingModel):
    nom = data_formatter.nomenclature_distance.nomenclature
    # clean data and put to indices
    cleaned_data = data_cleaner.clean_bi_df(nom, input_df)
    if not len(cleaned_data):
        return cleaned_data, []
    # get projections
    model_input = [cleaned_data[c].values.tolist() for c in input_columns]
    proj = project_input(*model_input, data_formatter=data_formatter, model=model)
    return cleaned_data, proj

def get_top_k_from_projection(cleaned_data, proj,
                              data_formatter: AnchorPositivePairsBatch, 
                              input_columns,
                              nb_top_values=10,
                              gt_column=None,
                              alpha_tree_mod=0,
                              beta_str_sim_mod=0):
    nom = data_formatter.nomenclature_distance.nomenclature
    similarity_func = SimilarityModel.similarity_func
    cleaned_main_field = cleaned_data[input_columns[0]].values.tolist()
    top_k_codes_and_similarities = nom.get_topk(proj, similarity_func, 
                                                nb_top_values=nb_top_values,
                                                alpha_tree=alpha_tree_mod,
                                                nom_distance=data_formatter.nomenclature_distance,
                                                beta_str_sim=beta_str_sim_mod,
                                                cleaned_inputs=cleaned_main_field)
    res = [{
        'cabbi': cleaned_data['cabbi'][i],
        'projection': proj[i],
        'top_k': {
            'codes': c,
            'similarities': s
        }
    } for i, (c, s) in enumerate(top_k_codes_and_similarities)]
    if gt_column is not None:
        gt_scores = nom.get_last_nodes_scores(cleaned_data[gt_column].values.tolist())
        for i in range(len(res)):
            res[i]['gt'] = {
                'code': cleaned_data[gt_column][i],
                'similarity': gt_scores[i][i]
            }
    return res

def get_top_k(input_df,
              input_columns,
              data_cleaner: Cleaner,
              data_formatter: AnchorPositivePairsBatch, 
              model: TrainingModel,
              nb_top_values=10,
              gt_column=None,
              alpha_tree_mod=0,
              beta_str_sim_mod=0):
    """
    calculer les top-k pour les données d'une dataframe
    IMPORTANT : le nombre de sorties peut être différent su nombre d'entrées car certains BI ont 
    des champs vides et ne sont pas traités
    --> faire la jointure sur le cabbi

    :param input_df: dataframe à processer
    :param input_columns: colonnes à récupérer dans input_df
    :param data_cleaner: NafCleaner utilisé à l'entrainement
    :param data_formatter: AnchorPositivePairsBatch utilisé à l'entrainement
    :param model: TrainingModel entrainé
    :param nb_top_values: nb de valeurs à renvoyer
    :param gt_column: colonne de Ground Truth (permet de récupérer le score de similarité de la GT)
    :param alpha_tree_mod: param pour le post-process basé sur la structure de la nomenclature
    :param beta_str_sim_mod: param pour le post-process basé sur la similarité textuelle
    :returns: liste de json
                {
                    'cabbi': cabbi du BI,
                    'top_k': {
                        'codes': liste de codes,
                        'similarities': liste de distances
                    },
                    'gt': {   #si gt_column
                        'code': gt_code,
                        'similarity': gt_score
                    }
                }
                où (liste de codes, liste de distances) ordonnées par distance décroissante
    """
    cleaned_data, proj = get_projection(input_df, input_columns, 
                                        data_cleaner, data_formatter, model)
    # logger.info(proj)
    # get topk k
    return get_top_k_from_projection(cleaned_data, proj,
                                     data_formatter,
                                     input_columns,
                                     nb_top_values,
                                     gt_column,
                                     alpha_tree_mod,
                                     beta_str_sim_mod)
    

def run_top_k_on_test(config, input_df, nb_docs, 
                      data_cleaner, data_formatter, model,
                      to_csv_file=None):
    """
    Calculer les performances du modèle en top-k sur l'ensemble des données de test
    
    :param config: configuration à appliquer
    :param input_df: itérable donnant des dataframes de données non préprocessées
    :param nb_docs: nombre total de documents testés
    :param data_cleaner: NafCleaner utilisé à l'entrainement
    :param data_formatter: AnchorPositivePairsBatch utilisé à l'entrainement
    :param model: TrainingModel entrainé
    :returns: tuple 
            - {
                  'nb_test_docs': nb_test_docs,
                  'nb_processed': voir doc de get_top_k,
                  'nb_not_processed': nb_not_processed,
                  'top_k': dict {k: nb docs avec gt dans top-k} pour k dans [1, 3, 5, 10],
                  'top_k_perc': {k: %age du test set avec gt dans top-k} pour k dans [1, 3, 5, 10]
              }
            - dataframe contenant les résultats de get_top_k pour tous les BI processés
    """
    logger = logging.getLogger('NomenclatureTopK')
    nb_tops = {
        1: 0,
        3: 0,
        5: 0,
        10: 0
    }
    nb_test_docs = 0
    batch_ind = 0
    batch_size = config['trainings']['training_params']['batch_size']
    nb_total = int(nb_docs/batch_size)
    input_columns = [c.replace('_repr', '_x') for c in config['trainings']['data']['input_columns']]
    gt_column = config['trainings']['data']['gt_column']
    alpha_tree_mod, beta_str_sim_mod = 0, 0
    if 'post_process' in config:
        if 'alpha_tree_mod' in config['post_process']:
            alpha_tree_mod = config['post_process']['alpha_tree_mod']
        if 'beta_str_sim_mod' in config['post_process']:
            beta_str_sim_mod = config['post_process']['beta_str_sim_mod']
    
    nb_not_processed = 0
    temp_result_file = to_csv_file if to_csv_file is not None else tempfile.NamedTemporaryFile(delete=False).name
    if os.path.exists(temp_result_file):
        os.remove(temp_result_file)
    os.makedirs(os.path.dirname(temp_result_file), exist_ok=True)
    try:
        for df in input_df:
            logger.info(f'Running batch {batch_ind}/{nb_total}')
            # logger.info(df)
            current_batch_size = len(df)
            top_k_codes_and_similarities = get_top_k(df, 
                                                     input_columns=input_columns,
                                                     data_cleaner=data_cleaner,
                                                     data_formatter=data_formatter, 
                                                     model=model,
                                                     nb_top_values=10,
                                                     gt_column=gt_column,
                                                     alpha_tree_mod=alpha_tree_mod,
                                                     beta_str_sim_mod=beta_str_sim_mod)
            # logger.info(top_k_codes_and_similarities)
            if gt_column:
                for top_10_codes in top_k_codes_and_similarities:
                    serie = df[df.cabbi == top_10_codes['cabbi']].iloc[0]
                    for k in nb_tops:
                        nb_tops[k] += serie[gt_column] in top_10_codes['top_k']['codes'][:k]
            nb_test_docs += len(top_k_codes_and_similarities)
            nb_not_processed += current_batch_size - len(top_k_codes_and_similarities)
            doc_results = [{
                'cabbi': r['cabbi'],
                'projection': r['projection'],
                'gt': r['gt']['code'] if gt_column else '',
                'gt_score': r['gt']['similarity'] if gt_column else '',
                'top_k_codes': r['top_k']['codes'],
                'top_k_similarities': r['top_k']['similarities']
            } for r in top_k_codes_and_similarities]
            batch_ind += 1
            # save to file
            with open(temp_result_file, 'a') as f:
                pd.DataFrame.from_dict(doc_results).to_csv(f, mode='a', header=f.tell()==0, index=False)
    except Exception as e:
        logger.error(f'Error calculating top-k : {e}')
        raise e
        
    if nb_test_docs == 0:
        logger.error(f'No test doc retrieved from BDD')
        raise RuntimeError('No test doc retrieved from BDD')

    # formattage des résultats
    results = {
        'nb_test_docs': nb_test_docs,
        'nb_processed': nb_test_docs - nb_not_processed,
        'nb_not_processed': nb_not_processed,
        'top_k': nb_tops,
        'top_k_perc': {k: v/float(nb_test_docs) for k, v in nb_tops.items()}
    }

    generic = lambda x: ast.literal_eval(x)
    conv = {'cabbi': str,
            'projection': generic,
            'top_k_codes': generic,
            'top_k_similarities': generic}
    result_df = pd.read_csv(temp_result_file, converters=conv)
    if to_csv_file is None:
        os.remove(temp_result_file)
    return results, result_df
    

if __name__ == "__main__":

    """
    Script permettant de calculer les performance d'un training en top-k

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
    TRAININGS_LOCAL_DIR, _ = get_trainings_dir(base_config)
    
    parser = argparse.ArgumentParser(
        description="Script pour calculer les performances top-k d'un modèle")
    parser.add_argument("model_dir", nargs='?', type=str, default=None,
                        help="dossier du training")
    parser.add_argument("weights_dir", nargs='?', type=str, default=None,
                        help="nom du dossier de poids à utiliser dans le training/train_weights")
    args = parser.parse_args()
    
    if hasattr(args, 'model_dir') and  args.model_dir:
        save_dir = os.path.abspath(args.model_dir)
    else:
        save_dir = get_last_local_training_dir(TRAININGS_LOCAL_DIR)
    if hasattr(args, 'weights_dir') and args.weights_dir:
        best_weights = os.path.abspath(args.weights_dir)
    else:
        best_weights = get_best_savedmodel(save_dir)

    # add log in training dir
    test_log_file = os.path.join(save_dir, 'topk.log')
    formatter = logging.Formatter(log_config['formatters']['simple']['format'])
    ch = logging.FileHandler(test_log_file)
    ch.setLevel('DEBUG')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger = logging.getLogger('NomenclatureTopK')
    
    logger.info(f'Running top-k pour {save_dir}/{best_weights}')
    config = load_config(save_dir)
    sync_with_minio = config['minio']['sync']

    # re-load all elements
    logger.info('Loading all trained objects')
    try:
        nomenclature, data_cleaner, data_formatter, model, _ = load_model_from_save_dir(save_dir, best_weights)
        modified = False
        if nomenclature.projections is None:
            logger.info('building nomenclature projections')
            nomenclature.build_nomenclature_projections(
                lambda x : model.run_model_single_side(
                    [np.array([emb], dtype=np.float32) for emb in data_formatter.format_input(x)])[0])
            modified = True
        if nomenclature.ngram_hot_encodings is None:
            nomenclature.create_trigram_repr()
            modified = True
        if modified:
            data_formatter.save(os.path.join(save_dir, "batcher"))
            if sync_with_minio:
                logger.info('pushing to minio')
                push_to_minio(save_dir)
    except Exception as e:
        logger.error(f'Error loading objects : {e}')
        exit(-1)

    # loading test data
    logger.info('Loading list of test BIs')
    try:
        test_cabbis = pd.read_csv(os.path.join(save_dir, 'cabbi_test.csv'))
        sql_request = config['data']['postgres_sql']
        cabbi_field = [t for t in sql_request.split(' ') if 'cabbi' in t][0].strip(',. ')
        list_of_cabbi = "('" + "','".join([str(v) for v in test_cabbis["cabbi"].values]) + "')"
        if 'where' in sql_request.lower():
            sql_request += ' AND '
        else:
            sql_request += ' WHERE '
        sql_request += f"{cabbi_field} IN {list_of_cabbi}"
    except Exception as e:
        logger.error(f'Error loading list of test BIs : {e}')
        exit(-1)

    # calculate values
    bdd = PostGre_SQL_DB()
    batch_size = config['trainings']['training_params']['batch_size']
    input_df = bdd.read_from_sql_with_chunksize(sql_request, batch_size)
    nb_docs = len(test_cabbis["cabbi"].values)
    logger.info('Calculating top-k')
    global_results, _ = run_top_k_on_test(config, input_df, nb_docs,
                                          data_cleaner, data_formatter, model,
                                          to_csv_file=os.path.join(save_dir, 'test_results.csv'))
    logger.info(f'Results: {global_results}')
    global_results['saved_model'] =  best_weights
    with open(os.path.join(save_dir, 'top_k.json'), 'w') as f:
        json.dump(global_results, f)
    
    # push on minio
    if sync_with_minio:
        push_to_minio(save_dir)