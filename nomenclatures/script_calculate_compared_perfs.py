#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#
# calculate_compared_perfs.py
#
#   /!\ IMPORTANT : Ce script dépend trop de l'organisation des données de la base pour être utilisable en ligne de commande
#
# Permet de comparer les résultats des systèmes actuels (MCA/Sicore) avec les nouveaux modèles
#
# il peut prendre en entrée pour les résultats des nouveaux modèles:
#   - une table générée par project_dbtable.py (plus rapide)
#   - un modèle entraîné (les résultats sont calculés à la volée, donc plus long)
#
# il génère:
#   - un json de statistiques sur 
#       - les résultats MCA et/ou SICORE
#       - les résultats des modèles actuels
#       - des comparaisons entre les 2
#   - un csv avec les valeurs d'input et les valeurs de des stats précédentes pour tous les BI
#       ayant des prédictions divergentes avec les 2 systèmes.
#
#
#
#   Pour le moment, gère NAF et PCS.
#   Pour ajouter une nomenclature, modifier get_gt_fieldname, get_sql_request, is_previously_coded
#   
#   Au moment de la création, les données des BI sont, pour chaque année, répartis entre
#   une table principale "rp_final_201X" et une table d'extraction des résultats intermédiaires de recap,
#   "profession_201X"
#   Si cela vient à changer, il faudra modifier get_sql_request
#
#
# @author cyril.poulet@starclay.fr
# @date: nov 2020

import sys
sys.path.append('..')
import os
import yaml
import json
import shutil
import logging.config
import s3fs
from tensorflow import keras
import numpy as np
import pandas as pd

from data_import.bdd import PostGre_SQL_DB
from data_import.csv_to_postgresql import read_table_description_file, import_csv_to_postegrsql
from training_utils import load_config, get_best_savedmodel
from script_run_top_k import load_model_from_save_dir, get_top_k


if __name__ == "__main__":

    # Logging - Chargement du fichier de configuration log
    with open(os.path.join(os.path.dirname(__file__), 'logging.conf.yaml'), 'r') as stream:
        log_config = yaml.load(stream, Loader=yaml.FullLoader)
    logging.config.dictConfig(log_config)

    logger = logging.getLogger()

    ##########################
    # Déclaration des inputs #
    ##########################
    input_dbtable_name = 'rp_final_2018'  # table principale
    input_prof_dbtable = 'profession_2018' # table secondaire

    # IMPORTANT : la table a précédence sur le save_dir
    # pour utiliser save_dir, il faut projections_dbtable_name=None
    projections_dbtable_name = 'pcs_projections_2018'  # table de résultats de project_dbtable.py
    save_dir = "trainings/PCS/2020-11-23_5"  # directory du modèle entrainé
    projections_dbtable_type = "pcs"   # nom de la nomenclature. "naf"|"pcs" (chargé auto si on charge un modèle)

    nb_psql_docs = None # nombre de docs à traiter (mettre à None pour traiter tous les docs de input_dbtable_name)

    if projections_dbtable_name is not None:
        save_dir = os.path.abspath('.')
        nomenclature_name = projections_dbtable_type
    else:
        best_weights = get_best_savedmodel(save_dir)
        config = load_config(save_dir)
        nomenclature_name = config['data']['nomenclature']['name'].lower()

    # add log in training dir
    test_log_file = os.path.join(save_dir, 'compared_perfs.log')
    formatter = logging.Formatter(log_config['formatters']['simple']['format'])
    ch = logging.FileHandler(test_log_file)
    ch.setLevel('DEBUG')
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger = logging.getLogger('ComparedPerformance')
    
    if projections_dbtable_name is not None:
        logger.info(f'Running compared perfs for result table {projections_dbtable_name}')
    else:
        logger.info(f'Running compared perfs for model {save_dir}/{best_weights}')


    def get_gt_fieldname():
        """
        Renvoie le nom du champ de ground_truth utilisé pour entrainé le modèle

        :returns: str
        """
        if nomenclature_name == 'naf':
            return 'actet_c'
        elif nomenclature_name == 'pcs':
            return 'prof_c'
        else:
            raise NotImplementedError(f'get_gt_fieldname not implemented for {nomenclature_name}')


    def get_sql_request(limit=None):
        """
        Crée la requète SQL
        Celle-ci doit contenir tous les champs nécessaires pour 
            - calculer une projection
            - calculer comment la solution actuelle a encodé le BI
            - si on passe par une table de projection, tous les champs de cette table

        :param limit: nb max de docs
        :returns: str, requète SQL
        """
        fields = [f'{input_dbtable_name}.cabbi', 'rs_x', 'actet_x', 'profs_x', 'profi_x', "profa_x"]
        if nomenclature_name == 'naf':
            fields += ['actet_c', 'actet_c_c'] + [f"{input_prof_dbtable}.{f}" for f in ['actet_c_m', 'i_mca_c', 'i_actet_c']]
        elif nomenclature_name == 'pcs':
            fields += [f"{input_prof_dbtable}.*"]
        else:
            raise NotImplementedError(f'get_sql_request not implemented for {nomenclature_name}')
        
        if projections_dbtable_name is not None:
            fields += [f"{projections_dbtable_name}.*"]

        req = f"SELECT " + ", ".join(fields)
        req += f" FROM {input_dbtable_name}"
        req += f" INNER JOIN {input_prof_dbtable}"
        req += f" ON {input_dbtable_name}.cabbi = {input_prof_dbtable}.cabbi"

        if projections_dbtable_name is not None:
            req += f" INNER JOIN {projections_dbtable_name}"
            req += f" ON {input_dbtable_name}.cabbi = {projections_dbtable_name}.cabbi"

        if limit is not None:
            req += f' LIMIT {limit}'
        return req


    def is_previously_coded(df_values):
        """
        Calcule un certain nombre d'indicateurs sur les systèmes actuels, pour chaque type de nomenclature

        :param df_value: dataframe obtenue avec get_sql_request. Si ce qu'il y a dans df_value ne permet pas de calculer 
                         le rang de la ground_truth (parce que get_gt_fieldname a été calculé), elle doit être enrichie
        :returns: df_value eventuellement enrichie, dataframe avec les indicateurs calculés
        """
        if nomenclature_name == "naf":
            coded = pd.DataFrame()
            coded['cabbi'] = df_values['cabbi']
            coded['mca_coded'] = df_values['i_mca_c'] == 'C'
            coded['sicore_coded'] = (df_values['i_mca_c'] == 'D') & (df_values['i_actet_c'] == 'CAC')
            coded['previous_auto_coded'] = coded['mca_coded'] | coded['sicore_coded']
            coded['previous_auto_impossible'] = df_values['i_actet_c'] == "XXX"
            coded['previous_auto_not_coded'] = ~(coded['previous_auto_coded'] | coded['previous_auto_impossible'])
            coded['previous_manual_coded'] = ~coded['previous_auto_coded'] & ~df_values['actet_c_m'].isna()
            coded['previous_not_coded'] = df_values['actet_c'].isna()
            return df_values, coded
        elif nomenclature_name == "pcs":
            # on aggrège les 2 types de profession
            prof = pd.DataFrame(columns=['cabbi', 'prof_c', 'prof_c_c', 'prof_c_m', 'i_prof_c'])
            prof['cabbi'] = df_values['cabbi']

            profs_inds = df_values.profs_c.notna()
            prof.loc[profs_inds, 'prof_c'] = df_values.loc[profs_inds, 'profs_c']
            prof.loc[profs_inds, 'prof_c_c'] = df_values.loc[profs_inds, 'profs_c_c']
            prof.loc[profs_inds, 'prof_c_m'] = df_values.loc[profs_inds, 'profs_c_m']
            prof.loc[profs_inds, 'i_prof_c'] = df_values.loc[profs_inds, 'i_profs_c']
            
            profi_inds = df_values.profs_c.isna() & df_values.profi_c.notna()
            prof.loc[profi_inds, 'prof_c'] = df_values.loc[profi_inds, 'profi_c']
            prof.loc[profi_inds, 'prof_c_c'] = df_values.loc[profi_inds, 'profi_c_c']
            prof.loc[profi_inds, 'prof_c_m'] = df_values.loc[profi_inds, 'profi_c_m']
            prof.loc[profi_inds, 'i_prof_c'] = df_values.loc[profi_inds, 'i_profi_c']
            
            profa_inds = df_values.profs_c.isna() & df_values.profi_c.isna()
            prof.loc[profa_inds, 'prof_c'] = df_values.loc[profa_inds, 'profa_c']
            prof.loc[profa_inds, 'prof_c_c'] = df_values.loc[profa_inds, 'profa_c_c']
            prof.loc[profa_inds, 'prof_c_m'] = df_values.loc[profa_inds, 'profa_c_m']
            prof.loc[profa_inds, 'i_prof_c'] = df_values.loc[profa_inds, 'i_profa_c']

            if 'prof_c' in df_values:
                df_values = df_values.drop(columns=['prof_c'])
            df_values = df_values.set_index('cabbi').join(prof.set_index('cabbi'), on='cabbi')
            df_values.reset_index(inplace=True)

            # on calcule les indicateurs
            def is_encoded_batch(x):
                return x is not None and (x[:2] == "CA" or x in ['AY2', 'AC2', 'AT2', 'AK2'])

            def is_encoded_recap_auto(x):
                return x is not None and (x[:2] == "CR" or x in ['RY2', "RC2", "RT2"])
            
            coded = pd.DataFrame()
            coded['cabbi'] = df_values['cabbi']
            coded['auto_coded_batch'] = df_values['prof_c_c'].notna() & df_values['i_prof_c'].map(is_encoded_batch)
            coded['auto_coded_recap'] = df_values['prof_c_c'].notna() & df_values['i_prof_c'].map(is_encoded_recap_auto)

            coded['previous_auto_coded'] = coded['auto_coded_batch'] | coded['auto_coded_recap']
            coded['previous_auto_not_coded'] = ~(coded['previous_auto_coded'])
            coded['previous_manual_coded'] = ~df_values['prof_c_m'].isna()
            coded['previous_not_coded'] = df_values['prof_c'].isna()
            return df_values, coded
        else:
            raise NotImplementedError(f'is_previously_encoded not available for {nomenclature_name}')


    def calculate_projection(nb_psql_docs=None):
        """
        Fonction qui charge le modèle, et génère les résultats DF par DF en calculant projections et top-k

        :param nb_sql_docs: nb de docs à traiter (paramètre de get_sql_request)
        :returns: yield data_df, result_dict avec 
                    data_df obtenu via get_sql_request
                    result_dict = {cabbi: {'cabbi': str,
                                            'projection': [float],
                                            'gt': str ou None,
                                            'gt_score': float ou None,
                                            'top_k_codes': [str],
                                            'top_k_similarities': [float],
                                            'codage_auto': bool
                                    }}
                    IMPORTANT: len(result_dict) peut être != de len(data_df) car certains cabbi ne peuvent être prédits
        """

        # re-load all elements
        logger.info('Loading all trained objects')
        try:
            nomenclature, data_cleaner, data_formatter, model, top1_classifier = load_model_from_save_dir(save_dir, best_weights)
            if nomenclature.projections is None:
                raise Exception('No nomenclature projection found. Please run run_top_k.py on model dir')
            if nomenclature.ngram_hot_encodings is None:
                raise Exception('No nomenclature trigram repr found. Please run run_top_k.py on model dir')

        except Exception as e:
            logger.error(f'Error loading objects : {e}')
            exit(-1)

        # calculate values
        bdd = PostGre_SQL_DB()
        sql_select_clause = get_sql_request(limit=nb_psql_docs)
        sql_count_request = f"SELECT COUNT(*) FROM {input_dbtable_name}"
        batch_size = config['trainings']['training_params']['batch_size']
        input_df = bdd.read_from_sql_with_chunksize(sql_select_clause, batch_size)
        nb_docs = bdd.read_from_sql(sql_count_request).values[0][0]
        logger.info('Calculating projections')
        
        nb_total = int(nb_docs/batch_size)
        input_columns = [c.replace('_repr', '_x') for c in config['trainings']['data']['input_columns']]
        gt_column = config['trainings']['data']['gt_column']
        alpha_tree_mod = config['post_process']['alpha_tree_mod']
        beta_str_sim_mod = config['post_process']['beta_str_sim_mod']
        batch_ind = 0

        try:
            for df in input_df:
                logger.info(f'Running batch {batch_ind}/{nb_total}')
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
                df_results = {r['cabbi']: {
                    'cabbi': r['cabbi'],
                    'projection': r['projection'],
                    'gt': r['gt']['code'] if gt_column else '',
                    'gt_score': r['gt']['similarity'] if gt_column else '',
                    'top_k_codes': r['top_k']['codes'],
                    'top_k_similarities': r['top_k']['similarities'],
                    'codage_auto': top1_classifier.validate_top1(r['top_k']['similarities']) if top1_classifier else False
                } for r in top_k_codes_and_similarities}
                batch_ind += 1
                yield df, df_results

        except Exception as e:
            logger.error(f'Error calculating top-k : {e}')
            raise e


    def load_projections(nb_psql_docs=None):
        """
        Fonction qui charge les projections via la table SQL de project_dbtable.py et les reformate au même format
        que calculate_projection

        :param nb_sql_docs: nb de docs à traiter (paramètre de get_sql_request)
        :returns: yield data_df, result_dict avec 
                    data_df obtenu via get_sql_request
                    result_dict = {cabbi: {'cabbi': str,
                                            'projection': [float],
                                            'gt': str ou None,
                                            'gt_score': float ou None,
                                            'top_k_codes': [str],
                                            'top_k_similarities': [float],
                                            'codage_auto': bool
                                    }}
                    IMPORTANT: len(result_dict) peut être != de len(data_df) car certains cabbi ne peuvent être prédits
        """
        bdd = PostGre_SQL_DB()
        sql_select_clause = get_sql_request(limit=nb_psql_docs)
        sql_count_request = f"SELECT COUNT(*) FROM {input_dbtable_name}"
        batch_size = 1000
        input_df = bdd.read_from_sql_with_chunksize(sql_select_clause, chunksize=batch_size)
        nb_docs = bdd.read_from_sql(sql_count_request).values[0][0]
        logger.info('loading projections')
        
        nb_total = int(nb_docs/batch_size)
        model_name = nomenclature_name
        batch_ind = 0

        try:
            for df in input_df:
                logger.info(f'Running batch {batch_ind}/{nb_total}')
                # logger.info(df)
                results_dict = {df['cabbi'].iloc[i]: {
                    'cabbi': df['cabbi'].iloc[i],
                    'projection': df[f"{model_name}_proj"].iloc[i],
                    'gt': '',
                    'gt_score': '',
                    'top_k_codes': [df[f"{model_name}_code_{j}"].iloc[i] for j in range(10)],
                    'top_k_similarities': [df[f"{model_name}_score_{j}"].iloc[i] for j in range(10)],
                    'codage_auto': df['codage_auto'].iloc[i] if "codage_auto" in df.columns else False
                } for i in range(len(df))}
                batch_ind += 1
                yield df, results_dict

        except Exception as e:
            pass

    
    # choix du provider
    projection_provider = calculate_projection
    if projections_dbtable_name :
        projection_provider = load_projections

    # Boucle principale
    gt_column = get_gt_fieldname()
    complete_results = None
    different_preds_file = os.path.join(save_dir, f'different_preds_{input_dbtable_name}.csv')
    if os.path.exists(different_preds_file):
        os.remove(different_preds_file)
    nb_docs = 0
    for df, results_dict in projection_provider(nb_psql_docs=nb_psql_docs):
        # get previous encoding (nomenclature dependant)
        df, df_results = is_previously_coded(df)

        # get current encoding
        gt_rangs = []
        for cabbi, res in results_dict.items():
            try:
                gt_rang = res['top_k_codes'].index(df[df.cabbi == cabbi].iloc[0][gt_column])
                gt_rangs.append([cabbi, gt_rang, res['codage_auto']])
            except ValueError:
                gt_rangs.append([cabbi, 10, res['codage_auto']])
        gt_rangs_df = pd.DataFrame(gt_rangs, columns=['cabbi', 'gt_rang', 'codage_auto'])
        gt_rangs_df['cabbi'] = gt_rangs_df['cabbi'].astype('object')
        df_results = df_results.set_index('cabbi').join(gt_rangs_df.set_index('cabbi'), on='cabbi')
        # here we take care of BI not predicted
        df_results.loc[df_results.codage_auto.isna(), 'codage_auto'] = False

        df_results['encoded_correctly'] = df_results['codage_auto'] & (df_results['gt_rang'] == 0)
        df_results['encoded_incorrectly'] = df_results['codage_auto'] & (df_results['gt_rang'] > 0)
        df_results['top1'] = df_results['gt_rang'] == 0
        df_results['top5'] = df_results['gt_rang'] < 5
        df_results['top10'] = df_results['gt_rang'] < 10
        df_results['not_found'] = df_results['gt_rang'] == 10
        df_results['not_calculated'] = df_results['gt_rang'].isna()

        # get compared encodings
        df_results['previous_auto_encoded_and_encoded_correctly'] = df_results['previous_auto_coded'] & df_results['encoded_correctly']
        df_results['previous_auto_encoded_and_encoded_incorrectly'] = df_results['previous_auto_coded'] & df_results['encoded_incorrectly']
        df_results['previous_auto_encoded_and_not_encoded'] = df_results['previous_auto_coded'] & ~df_results['codage_auto']
        df_results['previous_auto_encoded_and_top1'] = df_results['previous_auto_coded'] & df_results['top1']
        df_results['previous_auto_encoded_and_top5'] = df_results['previous_auto_coded'] & df_results['top5']
        df_results['previous_auto_encoded_and_top10'] = df_results['previous_auto_coded'] & df_results['top10']
        df_results['previous_auto_encoded_and_notfound'] = df_results['previous_auto_coded'] & df_results['not_found']
        df_results['previous_manual_encoded_and_encoded_correctly'] = df_results['previous_manual_coded'] & df_results['encoded_correctly']
        df_results['previous_manual_encoded_and_encoded_incorrectly'] = df_results['previous_manual_coded'] & df_results['encoded_incorrectly']
        df_results['previous_manual_encoded_and_not_encoded'] = df_results['previous_manual_coded'] & ~df_results['codage_auto']
        df_results['previous_manual_encoded_and_top1'] = df_results['previous_manual_coded'] & df_results['top1']
        df_results['previous_manual_encoded_and_top5'] = df_results['previous_manual_coded'] & df_results['top5']
        df_results['previous_manual_encoded_and_top10'] = df_results['previous_manual_coded'] & df_results['top10']
        df_results['previous_manual_encoded_and_notfound'] = df_results['previous_manual_coded'] & df_results['not_found']
        df_results['previous_not_encoded_and_encoded'] = df_results['previous_not_coded'] & df_results['codage_auto']
        df_results['previous_not_encoded_and_not_calculated'] = df_results['previous_not_coded'] & df_results['not_calculated']
        
        #we don't have the ground truth for previously not encoded cases, so no further analysis possible

        # aggregate counts
        if complete_results is None:
            complete_results = df_results.loc[:, df_results.columns  != 'cabbi'].sum(axis=0)
        else:
            complete_results += df_results.loc[:, df_results.columns != 'cabbi'].sum(axis=0)
        nb_docs += len(df)

        # keep results that are different 
        df = df.drop(columns=['codage_auto'])      # doublon
        df = df.set_index('cabbi').join(df_results, on='cabbi')
        df.reset_index(inplace=True)
        different_preds = df[~df['previous_auto_encoded_and_top1'] 
                             & ~df['previous_manual_encoded_and_top1']
                             & ~df['previous_not_encoded_and_not_calculated']]
        with open(different_preds_file, 'a') as f:
                different_preds.to_csv(f, mode='a', sep=";", header=f.tell()==0, index=False)

    # save results
    logger.info('Saving results')
    complete_results = complete_results.drop('gt_rang')
    complete_results /= nb_docs
    complete_results['nb_docs'] = nb_docs
    with open(os.path.join(save_dir, f'complete_comp_results_{input_dbtable_name}.json'), 'w') as f:
        json.dump(complete_results.to_dict(), f)