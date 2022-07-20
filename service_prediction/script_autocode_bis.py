from datetime import datetime
from pprint import pprint
import json
import logging
import sys

import pandas as pd
#import numpy as np
#from multiprocessing import  Pool

from src.config import *
from src.data_import.bdd import PostGre_SQL_DB
from api_functions import *

"""
def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df
"""
#train = parallelize_dataframe(train_df, add_features)

def integrate_siret_results(bis, api_results, complete_coding_results, sirus_table, my_driver_PostGre):
    """
    Update des BI et des résultats du codage avec les prédictions SIRET
    + récupération du code apet associé

    mapping : 
        bi.siretc = siret de prediction[0]
        bi.i_siret_c = prediction['codage_auto'] (bool)
        bi.actet_c = apet de l'établissement correspondant à bi.siretc
        bi.i_actet_c = prediction['codage_auto'] (bool)

    :param bis: [pd.Series] (BIs à prédire)
    :param api_results: dict résultat de l'appel API SIRET
    :param complete_coding_results: dict {cabbi: {...}} contenant l'ensemble des résultats de prédictions
    :param sirus_table: table établissement à requeter
    :param my_driver_PostGre: objet PostGre_SQL_DB
    :return: bis, complete_coding_results   (même formats, updatés)
    """
    api_results = {res['cabbi']: res for res in api_results['siret']}
    for bi in bis:
        result = api_results[bi.cabbi]
        if result['status'] == 'ok':
            bi.siretc = result['predictions'][0][0]
            bi.i_siret_c = int(result['codage_auto'])
            complete_coding_results[bi.cabbi]['siret_predictions'] = result['predictions']
            complete_coding_results[bi.cabbi]['siret'] = result['predictions'][0][0]
            complete_coding_results[bi.cabbi]['siret_score'] = result['predictions'][0][1]
            complete_coding_results[bi.cabbi]['siret_codage_auto'] = result['codage_auto']
            complete_coding_results[bi.cabbi]['siret_codage_auto_proba'] = result['codage_auto_proba']
            complete_coding_results[bi.cabbi]['siret_error'] = ''
            complete_coding_results[bi.cabbi]['siret_apet'] = ""
        else:
            complete_coding_results[bi.cabbi]['siret_predictions'] = []
            complete_coding_results[bi.cabbi]['siret'] = ''
            complete_coding_results[bi.cabbi]['siret_score'] = 0
            complete_coding_results[bi.cabbi]['siret_codage_auto'] = False
            complete_coding_results[bi.cabbi]['siret_codage_auto_proba'] = 0
            complete_coding_results[bi.cabbi]['siret_error'] = result['error']
            complete_coding_results[bi.cabbi]['siret_apet'] = ""

    sirets = set([el["siret"] for el in complete_coding_results.values() if el['siret']])
    if len(sirets):
        sql_apet = f'SELECT CONCAT(sirus_id, nic) AS siret, apet FROM {sirus_table} WHERE sirus_id || nic IN '
        if len(sirets) > 1:
            sql_apet += str(tuple(sirets))
        else :
            sql_apet += "('" + list(sirets)[0] + "')"
        df = my_driver_PostGre.read_from_sql(sql_apet)

        for bi in bis:
            if len(df[df.siret == bi.siretc]):
                bi.actet_c = df[df.siret == bi.siretc].apet.values[0]
                bi.i_actet_c = bi.i_siret_c
                complete_coding_results[bi.cabbi]['siret_apet'] = bi.actet_c

    return bis, complete_coding_results


def integrate_naf_results(bis, api_results, complete_coding_results):
    """
    Update des BI et des résultats du codage avec les prédictions NAF

    mapping : PAS DE MODIFICATION DES BIs

    :param bis: [pd.Series] (BIs à prédire)
    :param api_results: dict résultat de l'appel API SIRET
    :param complete_coding_results: dict {cabbi: {...}} contenant l'ensemble des résultats de prédictions
    :return: bis, complete_coding_results   (même formats, updatés)
    """
    api_results = {res['cabbi']: res for res in api_results['naf']}
    for cabbi, coding_res in complete_coding_results.items():
        result = api_results[cabbi]
        if result['status'] == 'ok':
            coding_res['naf'] = result['predictions'][0][0]
            coding_res['naf_score'] = result['predictions'][0][1]
            coding_res['naf_predictions'] = result['predictions']
            coding_res['naf_codage_auto'] = result['codage_auto']
            coding_res['naf_codage_auto_proba'] = result['codage_auto_proba']
            coding_res['naf_error'] = ''
        else:
            coding_res['naf'] = ''
            coding_res['naf_score'] = 0
            coding_res['naf_predictions'] = []
            coding_res['naf_codage_auto'] = False
            coding_res['naf_codage_auto_proba'] = 0
            coding_res['naf_error'] = result['error']

    return bis, complete_coding_results


def integrate_pcs_results(bis, api_results, complete_coding_results):
    """
    Update des BI et des résultats du codage avec les prédictions PCS

    mapping : 
        bi.prof(s/a/i)_c = pcs de prediction[0]
        bi.i_prof(s/a/i)_c = prediction['codage_auto'] (bool)

    :param bis: [pd.Series] (BIs à prédire)
    :param api_results: dict résultat de l'appel API SIRET
    :param complete_coding_results: dict {cabbi: {...}} contenant l'ensemble des résultats de prédictions
    :return: bis, complete_coding_results   (même formats, updatés)
    """
    api_results = {res['cabbi']: res for res in api_results['pcs']}
    for bi in bis:
        result = api_results[bi.cabbi]
        if result['status'] == 'ok':
            if bi.profs_x : prof_field = 'profs_c'
            elif bi.profi_x : prof_field = 'profi_c'
            else : prof_field = 'profa_c'
            bi[prof_field] = result['predictions'][0][0]
            bi['i_'+prof_field] = int(result['codage_auto'])
            
            complete_coding_results[bi.cabbi]['pcs_predictions'] = result['predictions']
            complete_coding_results[bi.cabbi]['pcs_field'] = prof_field
            complete_coding_results[bi.cabbi]['pcs'] = result['predictions'][0][0]
            complete_coding_results[bi.cabbi]['pcs_score'] = result['predictions'][0][1]
            complete_coding_results[bi.cabbi]['pcs_codage_auto'] = result['codage_auto']
            complete_coding_results[bi.cabbi]['pcs_codage_auto_proba'] = result['codage_auto_proba']
            complete_coding_results[bi.cabbi]['pcs_error'] = ''
        else:
            complete_coding_results[bi.cabbi]['pcs_predictions'] = []
            complete_coding_results[bi.cabbi]['pcs_field'] = ''
            complete_coding_results[bi.cabbi]['pcs'] = ''
            complete_coding_results[bi.cabbi]['pcs_score'] = 0
            complete_coding_results[bi.cabbi]['pcs_codage_auto'] = False
            complete_coding_results[bi.cabbi]['pcs_codage_auto_proba'] = 0
            complete_coding_results[bi.cabbi]['pcs_error'] = result['status']
    return bis, complete_coding_results


def integrate_noncodable_results(bis, api_results, complete_coding_results):
    """
    Update des BI et des résultats du codage avec les prédictions noncodables

    mapping : PAS DE MODIFICATION DES BIs

    :param bis: [pd.Series] (BIs à prédire)
    :param api_results: dict résultat de l'appel API SIRET
    :param complete_coding_results: dict {cabbi: {...}} contenant l'ensemble des résultats de prédictions
    :return: bis, complete_coding_results   (même formats, updatés)
    """
    api_results = {res['cabbi']: res for res in api_results['noncodable']}
    for cabbi, coding_res in complete_coding_results.items():
        result = api_results[cabbi]
        if result['status'] == 'ok':
            coding_res['non_codable'] = result['non_codable']
            coding_res['non_codable_score'] = result['predictions'][1]
            coding_res['non_codable_error'] = ''
        else:
            coding_res['non_codable'] = False
            coding_res['non_codable_score'] = 0
            coding_res['non_codable_error'] = result['status']
    return bis, complete_coding_results


def push_updated_bis(bis, bi_table, sql_bdd):
    """
    Insertion des BIs mis à jours dans une table de la bdd

    :param bis: [pd.Series] (BIs à prédire)
    :param bi_table: table dans laquelle insérer (ATTENTION: pas d'update possible...)
    :param sql_bdd: PostGre_SQL_DB
    :return: None
    """
    global new_bi_table_created
    df = pd.DataFrame(bis)
    df.set_index('cabbi')
    df.to_sql(bi_table,
              sql_bdd.engine,
              index=False,
              if_exists='append')
    if not new_bi_table_created:
        with sql_bdd.engine.connect() as con:
            con.execute(f"ALTER TABLE {bi_table} ADD PRIMARY KEY (cabbi);")
        new_bi_table_created = True


def push_complete_results(complete_results, results_table, sql_bdd):
    """
    Insertion des résultats complets dans une table de la bdd

    :param complete_results: dict {cabbi: {...}}
    :param results_table: table dans laquelle insérer (ATTENTION: pas d'update possible...)
    :param sql_bdd: PostGre_SQL_DB
    :return: None
    """
    global result_table_created
    complete_results = [{'cabbi': k, **v} for k, v in complete_results.items()]
    df = pd.DataFrame.from_dict(complete_results)
    df.set_index('cabbi')
    df.siret_predictions = df.siret_predictions.apply(lambda x: json.dumps(x))
    df.naf_predictions = df.naf_predictions.apply(lambda x: json.dumps(x))
    df.pcs_predictions = df.pcs_predictions.apply(lambda x: json.dumps(x))
    df.to_sql(results_table,
            sql_bdd.engine,
            index=False,
            if_exists='append')
    if not result_table_created:
        with sql_bdd.engine.connect() as con:
            con.execute(f"ALTER TABLE {results_table} ADD PRIMARY KEY (cabbi);")
        result_table_created = True


def process_bis(bis, adresse_api, api_session, sirus_table, bi_output_table, results_table, sql_bdd):
    """
    Fonction complète d'appels du service de prédiction, exploitation des résultats et push dans la db

    :param bis: [pd.Series] (BIs à prédire)
    :param adresse_api: str, adresse de l'API
    :param api_session: request.session portant le token d'identification
    :param sirus_table: table établissement
    :param bi_output_table: table dans laquelle insérer les BI updatés (ATTENTION: pas d'update possible...)
    :param results_table: table dans laquelle insérer les résultats complets (ATTENTION: pas d'update possible...)
    :param sql_bdd: PostGre_SQL_DB
    :return: None
    """
    docs_for_call = [format_input(bi) for bi in bis]
    complete_coding_results = {bi.cabbi: {} for bi in bis}
    siret_results = predict_siret(adresse_api, api_session, docs_for_call)
    naf_results = predict_naf(adresse_api, api_session, docs_for_call)
    pcs_results = predict_pcs(adresse_api, api_session, docs_for_call)
    non_codable_results = predict_noncodable(adresse_api, api_session, docs_for_call)

    bis, complete_coding_results = integrate_siret_results(bis, siret_results, complete_coding_results, sirus_table, sql_bdd)
    bis, complete_coding_results = integrate_naf_results(bis, naf_results, complete_coding_results)
    bis, complete_coding_results = integrate_pcs_results(bis, pcs_results, complete_coding_results)
    bis, complete_coding_results = integrate_noncodable_results(bis, non_codable_results, complete_coding_results)    
    push_updated_bis(bis, bi_output_table, sql_bdd)
    push_complete_results(complete_coding_results, results_table, sql_bdd)


if __name__ == "__main__":
    """
    Script permettant de passer le codage automatique sur une table RP en appelant le service prédiction
    (en local ou déployé)

    le script:
        - update les BI avec les prédictions et les push dans une NOUVELLE table
            - prédiction SIRET:
                bi.siretc = siret de prediction[0]
                bi.i_siret_c = prediction['codage_auto'] (int)
                bi.actet_c = apet de l'établissement correspondant à bi.siretc
                bi.i_actet_c = prediction['codage_auto'] (int)
            - prédiction PCS:
                bi.prof(s/a/i)_c = pcs de prediction[0]
                bi.i_prof(s/a/i)_c = prediction['codage_auto'] (int)

        - créée une nouvelle table contenant l'ensemble des résultats de prédictions SIRET, NAF, PCS et non codables
    
    Pour être lancé, vous devez avoir lancé avant un des script build_docker_image...sh
    """

    adresse_api = "http://localhost:8000"
    #adresse_api = "https://aiee2-prediction-api.ouest.innovation.insee.eu"

    api_login = "admin_recap"
    api_password = "admin_recap"

    nb_docs_by_call = 30  # nb bis par batch. max 15 à distance

    bi_input_table = "rp_2020_todo"                # table à traiter
    bi_output_table = "rp_2020_output_testasync"          # table à créer (update de bi_input_table)
    sirus_table = "sirus_2020"                      # table établissement
    results_table = "rp_2020_codageauto_results_testasync"    # table de résultats à créer

    sql_bi: str = f"SELECT * FROM {bi_input_table}" 
      
    # identification et récupération du token
    api_session = create_session(adresse_api, api_login, api_password)

    my_driver = PostGre_SQL_DB(host=db_host, port=db_port, dbname=db_dbname, user=db_user, password=db_password)
    
    # indicateurs de création de table (utilisé pour insérer la clé cabbi dans les tables)  
    sql_new_bi_table_created: str = f"SELECT EXISTS(SELECT 1 FROM pg_constraint WHERE conname = '{bi_output_table}_pkey')"
    with my_driver.engine.connect() as con:
        r=con.execute(sql_new_bi_table_created)
    new_bi_table_created=r.next()[0]
    sql_result_table_created: str = f"SELECT EXISTS(SELECT 1 FROM pg_constraint WHERE conname = '{results_table}_pkey')"
    with my_driver.engine.connect() as con:
        r=con.execute(sql_result_table_created)
    result_table_created=r.next()[0]
    
    nb_bis = 0
    current_bis = []
    errors = {}
    start_time = datetime.now()
    for df in my_driver.read_from_sql_with_chunksize(sql_bi, chunksize=1000):
        for i, bi in df.iterrows():
            current_bis.append(bi)
            if len(current_bis) == nb_docs_by_call:
                try:
                    print(f'processing call {nb_bis // nb_docs_by_call}')
                    process_bis(current_bis, adresse_api, api_session, sirus_table, bi_output_table, results_table, my_driver)
                except Exception as e:
                    print(e)
                    #print(current_bis)
                    #exit(-1)        # On ne veut pas quitter, on veut continuer à traiter le reste de la requète
                nb_bis += len(current_bis)
                current_bis = []

    if len(current_bis):
        print(f'processing call {nb_bis // nb_docs_by_call}')
        process_bis(current_bis, adresse_api, api_session, sirus_table, bi_output_table, results_table, my_driver)

    total_time = datetime.now() - start_time

    results = {
        "total_time": str(total_time),
        "nb_bis": nb_bis,
        "errors": errors
    }

    pprint(results)
    with open(f'log_autocodage_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
        json.dump(results, f)
        
        
