"""
Récupération de code depuis https://git.stable.innovation.insee.eu/innovation/AIEE/-/blob/master/enrichissement/stephanie/sirus_sigles.ipynb

Génération de sigle à partir de sirus 2017


Author : bsanchez@starclay.fr
date : 16/07/2020
"""

import os
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import psycopg2
import yaml
import time
from tqdm import tqdm
import io
from stop_words import get_stop_words
import string
import re
from bdd import *


my_driver = PostGre_SQL_DB()


def main():
    test = my_driver.read_from_sql("""
    SELECT sirus_id,nic,enseigne_et1,adr_et_l1,adr_et_l2 
    FROM sirus_2017
    """)
    denomination = test.loc[test.adr_et_l1.isna()==False,["sirus_id","nic","adr_et_l1"]]
# Suppression des champs débutant par monsieur madame dans l'adresse (nom de personnes et pas raison sociale d'entreprise)
    selection = [w.strip().split(' ')[0] not in ['MONSIEUR','MADAME'] for w in denomination.adr_et_l1]
    denomination = denomination.loc[selection,:]
    stop_words = get_stop_words('french')
    stops = stop_words.append([])
    words = list(set(denomination.adr_et_l1))
    sortie = []

    for s in words:
        o = re.sub("-"," ",s)
        o = o.split(" ")
        out = [re.sub(r"[\w]{1}'",'',w) for w in o]
        out = [re.sub(r'[^\w\s]','',w) for w in out]
        out = [w[:1] for w in out if (w.lower() not in stop_words) ]
        out = ''.join(out)
        sortie.append(out)
    
    table = pd.DataFrame({"adr_et_l1":words,"sigle":sortie})
    export = pd.merge(denomination,table,on="adr_et_l1",how='outer')
    print(export.shape)
    print(denomination.shape)
    export.to_csv("sigles_adr_et_l1_sirus.csv",sep=",")

if __name__ == "__main__":
    main()