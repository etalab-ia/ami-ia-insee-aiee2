import sys
import os
sys.path.insert(0, os.path.abspath('..'))
import json
from datetime import datetime
from pprint import pprint

from src.data_import.bdd import PostGre_SQL_DB
from ..api_functions import *


def analyse_results(input_bis, prediction_results, topks, errors):
    prediction_results = {str(r['cabbi']): r for r in prediction_results['naf']}
    prediction_status = {k: v['status'] for k, v in prediction_results.items()}
    predictions = {k: v['predictions'] for k, v in prediction_results.items() if 'predictions' in v}
    for input_bi in input_bis:
        if prediction_status[input_bi.cabbi] == 'error':
            errors[input_bi.cabbi] = prediction_results[input_bi.cabbi]['error']
            continue
        preds = [p[0] for p in predictions[input_bi.cabbi]]
        try:
            ind = preds.index(input_bi.actet_c)
            for key in topks:
                if key == 'not_found':
                    continue
                if ind < key:
                    topks[key] += 1
        except:
            topks['not_found'] += 1
    return topks, errors


if __name__ == "__main__":

    adresse_api = "http://localhost:8000"
    adresse_api = "https://aiee2-prediction-api.ouest.innovation.insee.eu"

    nb_docs_by_call = 50

    s = create_session(adresse_api, "admin_recap", "admin_recap2")

    my_driver = PostGre_SQL_DB()
    sql_bi: str = f"SELECT * FROM rp_final_2019 WHERE NOT (actet_x IS NULL OR actet_c IS NULL) LIMIT 100;" # WHERE cabbi = '9306136814'
    nb_bis = 0
    current_bis = []
    docs_for_call = []
    topks = {1: 0, 2: 0, 3: 0, 5: 0, 10: 0, 'not_found': 0}
    errors = {}
    start_time = datetime.now()
    for df in my_driver.read_from_sql_with_chunksize(sql_bi, chunksize=1000):
        for i, bi in df.iterrows():
            docs_for_call.append(format_input(bi))
            current_bis.append(bi)
            if len(docs_for_call) == nb_docs_by_call:
                try:
                    print(f'processing call {nb_bis // nb_docs_by_call}')
                    res = predict_naf(adresse_api, s, docs_for_call)
                except Exception as e:
                    print(e)
                    pprint(docs_for_call)
                    exit(-1)
                topks, errors = analyse_results(current_bis, res, topks, errors)
                nb_bis += len(docs_for_call)
                docs_for_call = []
                current_bis = []

    if len(docs_for_call):
        res = predict_naf(adresse_api, s, docs_for_call)
        topks, errors = analyse_results(current_bis, res, topks, errors)
        nb_bis += len(docs_for_call)

    total_time = datetime.now() - start_time

    results = {
        "total_time": str(total_time),
        "topks": topks,
        "topks_pct": {k: v/float(max(nb_bis - len(errors), 1)) for k, v in topks.items()},
        "nb_bis": nb_bis,
        "errors": errors
    }

    pprint(results)
    with open(f'test_naf_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
        json.dump(results, f)