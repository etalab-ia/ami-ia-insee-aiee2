import sys
import os
sys.path.insert(0, os.path.abspath('..'))
import json
from datetime import datetime
from pprint import pprint

from src.data_import.bdd import PostGre_SQL_DB
from ..api_functions import *


def analyse_results(input_bis, prediction_results, tp, tn, fp, fn):
    prediction_results = prediction_results['noncodable']
    prediction_results = {r['cabbi']: r['non_codable'] for r in prediction_results}
    for input_bi in input_bis:
        has_siret = any([input_bi.siretc, input_bi.siretm, input_bi.siret_dec])
        if has_siret:
            if prediction_results[input_bi.cabbi] :
                fp += 1
            else:
                tn += 1
        else:
            if prediction_results[input_bi.cabbi] :
                tp += 1
            else:
                fn += 1

    return tp, tn, fp, fn


if __name__ == "__main__":

    adresse_api = "http://localhost:8000"
    adresse_api = "https://aiee2-prediction-api.ouest.innovation.insee.eu"

    nb_docs_by_call = 50

    s = create_session(adresse_api, "admin_recap", "admin_recap2")

    my_driver = PostGre_SQL_DB()
    sql_bi: str = f"SELECT * FROM rp_final_2019 LIMIT 1000;" #WHERE cabbi = '9350082027' 
    nb_bis = 0
    current_bis = []
    docs_for_call = []
    tp, tn, fp, fn = 0, 0, 0, 0
    start_time = datetime.now()
    for df in my_driver.read_from_sql_with_chunksize(sql_bi, chunksize=1000):
        for i, bi in df.iterrows():
            docs_for_call.append(format_input(bi))
            current_bis.append(bi)
            if len(docs_for_call) == nb_docs_by_call:
                try:
                    print(f'processing call {nb_bis // nb_docs_by_call}')
                    res = predict_noncodable(adresse_api, s, docs_for_call)
                except Exception as e:
                    print(e)
                    pprint(docs_for_call)
                    exit(-1)
                tp, tn, fp, fn = analyse_results(current_bis, res, tp, tn, fp, fn)
                nb_bis += len(docs_for_call)
                docs_for_call = []
                current_bis = []

    if len(docs_for_call):
        res = predict_noncodable(adresse_api, s, docs_for_call)
        tp, tn, fp, fn = analyse_results(current_bis, res, tp, tn, fp, fn)

    total_time = datetime.now() - start_time
    precision = tp / float(tp + fp) if tp + fp > 0 else 1
    recall = tp / float(tp + fn) if tp + fn > 0 else 1
    f1 = 2 * precision * recall / float(precision + recall)
    results = {
        "total_time": str(total_time),
        "tp": tp, 
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "nb_bis": tp + tn + fp + fn,
        "nb_bis_positive": tp + fn,
        "nb_bis_negative": tn + fp
    }

    pprint(results)
    with open(f'test_noncodable_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
        json.dump(results, f)