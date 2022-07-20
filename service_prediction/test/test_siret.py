import sys
import os
sys.path.insert(0, os.path.abspath('..'))
import json
from datetime import datetime
from pprint import pprint

from src.data_import.bdd import PostGre_SQL_DB
from ..api_functions import *


def analyse_results(input_bis, prediction_results, topks, tp, tn, fp, fn):
    prediction_results = prediction_results['siret']
    siret_results = {str(r['cabbi']): r['predictions'] for r in prediction_results}
    for input_bi in input_bis:
        preds = [p[0] for p in siret_results[input_bi.cabbi]]
        try:
            ind = preds.index(input_bi.siretc)
            for key in topks:
                if key == 'not_found':
                    continue
                if ind < key:
                    topks[key] += 1
        except:
            topks['not_found'] += 1
    
    codage_auto_results =  {str(r['cabbi']): r['codage_auto'] for r in prediction_results}
    for input_bi in input_bis:
        top1_pred = siret_results[input_bi.cabbi][0][0]
        top1_correct = (top1_pred == input_bi.siretc)
        if top1_correct:
            if codage_auto_results[input_bi.cabbi]:
                tp += 1
            else:
                fn += 1
        else:
            if codage_auto_results[input_bi.cabbi]:
                fp += 1
            else:
                tn += 1
    return topks, tp, tn, fp, fn


if __name__ == "__main__":

    adresse_api = "http://localhost:8000"
    adresse_api = "https://aiee2-prediction-api.ouest.innovation.insee.eu"

    nb_docs_by_call = 10

    s = create_session(adresse_api, "admin_recap", "admin_recap2")

    my_driver = PostGre_SQL_DB()
    sql_bi: str = "SELECT * FROM rp_final_2019 WHERE siretc IS NOT NULL LIMIT 100"
    nb_bis = 0
    current_bis = []
    docs_for_call = []
    topks = {1: 0, 2: 0, 3: 0, 5: 0, 10: 0, 'not_found': 0}
    tp, tn, fp, fn = 0, 0, 0, 0

    start_time = datetime.now()
    for df in my_driver.read_from_sql_with_chunksize(sql_bi, chunksize=1000):
        for i, bi in df.iterrows():
            # print(bi.cabbi)
            ## décommentez la ligne suivante pour tester le comportement sur une requète "manuelle", ie un cabbi non connu du système
            # bi['cabbi'] = 'test'
            docs_for_call.append(format_input(bi))
            current_bis.append(bi)
            if len(docs_for_call) == nb_docs_by_call:
                try:
                    print(f'processing call {nb_bis // nb_docs_by_call}')
                    res = predict_siret(adresse_api, s, docs_for_call)
                except Exception as e:
                    print(e)
                    pprint(docs_for_call)
                    exit(-1)
                topks, tp, tn, fp, fn = analyse_results(current_bis, res, topks, tp, tn, fp, fn)
                nb_bis += len(docs_for_call)
                docs_for_call = []
                current_bis = []

    if len(docs_for_call):
        res = predict_siret(adresse_api, s, docs_for_call)
        topks, tp, tn, fp, fn = analyse_results(current_bis, res, topks, tp, tn, fp, fn)
        nb_bis += len(docs_for_call)

    total_time = datetime.now() - start_time
    codage_precision = tp / float(tp + fp) if tp + fp > 0 else 1
    codage_recall = tp / float(tp + fn) if tp + fn > 0 else 1
    codage_f1 = 2 * codage_precision * codage_recall / float(codage_precision + codage_recall)

    results = {
        "total_time": str(total_time),
        "topks": topks,
        "topks_pct": {k: v/float(nb_bis) for k, v in topks.items()},
        "codage_tp": tp, 
        "codage_tn": tn,
        "codage_fp": fp,
        "codage_fn": fn,
        "codage_precision": codage_precision,
        "codage_recall": codage_recall,
        "codage_f1": codage_f1,
        "nb_bis_codables": tp + fn,
        "nb_bis_noncodables": tn + fp,
        "nb_bis": nb_bis
    }

    pprint(results)
    with open(f'test_siret_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
        json.dump(results, f)