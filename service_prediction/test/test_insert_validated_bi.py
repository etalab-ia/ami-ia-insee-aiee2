import sys
import os
sys.path.insert(0, os.path.abspath('..'))
import json
from datetime import datetime
from pprint import pprint

from src.data_import.bdd import PostGre_SQL_DB
from ..api_functions import *


def format_input(input_dict):
    j = {
        "cabbi": input_dict.get("cabbi"),
    }
    return j


if __name__ == "__main__":
    # ATTENTION : ce test va modifier le contenu d'ES sur lequel le service est branché...

    adresse_api = "http://localhost:8000"
    # adresse_api = "https://aiee2-prediction-api.ouest.innovation.insee.eu"

    nb_docs_by_call = 1

    s = create_session(adresse_api, "admin_recap", "admin_recap")

    my_driver = PostGre_SQL_DB()
    sql_bi: str = f"SELECT rp_final_2019.cabbi FROM rp_final_2019 WHERE cabbi = '9302969851'"
    sql_bi += " LIMIT 100;"
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
                    res = call_prediction(adresse_api, "save_bi", s, docs_for_call)
                except Exception as e:
                    print(e)
                    pprint(docs_for_call)
                    exit(-1)

    total_time = datetime.now() - start_time