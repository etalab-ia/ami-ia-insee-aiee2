"""
Ficher contenant une classe driver qui permette de communiquer avec le cluster ElasticSearch.
Charge les requetes, les execute et formatte les résultats

https://elasticsearch-py.readthedocs.io/en/7.9.1/api.html

2020/11/19
"""

# !pip install elasticsearch
import sys
sys.path.append("..")
import csv
import asyncio
from elasticsearch import AsyncElasticsearch, Elasticsearch
from elasticsearch.helpers import parallel_bulk
import s3fs
import time
import os
from collections import deque
import json
import pandas as pd
import data_import.bdd as bdd
import glob
import collections
#A virer après tests
from config import *

def load_queries(dir):
    """
    Charge les requètes depuis des fichiers json dans :param dir:
    """
    dict={}
    for file in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, file)):
            with open(os.path.join(dir, file)) as f:
                query=[]
                for line in f:
                    query.append(line)
                dict[file]=''.join(query)
                
def genreq():
    """
    Génère une requète test depuis test.csv
    """
    with open("test.csv",'r',encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile, delimiter=",")
        next(reader)
        for row in reader:
            identent="5"
            siren_c=""
            if siren_c=="":
                yield identent,{
                    "id": "my_script", #id_requete,
                    "params": {
                        "RS_X": "NATIXIS"
                        }
                    }
                
def gen_data_from_file(data, index):
    """
    Generateur avec specification de l'id
    
    :param data: pd.DataFrame, donnée à exporter vers le ES
    :param index: requete json contenant le mapping ES
    """
    if "sirus" in index:
        for idx, row in data.iterrows():
            header= {"_op_type": "index" # https://elasticsearch-py.readthedocs.io/en/7.9.1/helpers.html
                        , "_index": index
                        , "_id": row["sirus_id"] + row["nic"]}
            yield {**header,**row}
    if "rp" in index:
        for idx, row in data.iterrows():
            header= {"_op_type": "index" # https://elasticsearch-py.readthedocs.io/en/7.9.1/helpers.html
                        , "_index": index
                        , "_id": row["cabbi"]}
            yield {**header,**row}

                
class ElasticDriver():
    
    def __init__(self, host, port, http_proxy=None, requests_json_dir='elastic'):
        """
        Classe de driver ES

        :param host: adresse du cluster (str ou liste de str)
        :param port: port à utiliser
        :param http_proxy: non utilisé (si proxy, le passer dans les variables d'env http_proxy et https_proxy)
        :param requests_json_dir: dossier contenant les définitions des requètes en json
        """
    
        self.HOST: str = host
        self.port: str = port
            
        self.elastic = Elasticsearch(self.HOST, maxsize=64, http_compress=True, timeout=120)
        self.requests_json_dir = requests_json_dir
        
        
    def load_query(self, name: str):
        """
        Chargement d'un fichier json dans le dossier elastic
        
        :param name: nom du fichier.json
        :returns: requete en json
        """
        with open(os.path.join(self.requests_json_dir, name + '.json')) as json_file:
            query = json.load(json_file)
        return query
    
    def request(self, name_query: str, index: str, var: list = None):
        """
        Execute un requete ES
        
        :param name_query: nom de le requete à executer
        :param index: nom de la table
        :param var: list des 'params'
        :returns: list, list de hits
        """
        nb_params: int = len(var)
           
        body_header = json.dumps({"index" : index})+ "\n"
        query_dict = self.load_query(name_query)
        
        body_chunk = ""
        for i in range(nb_params):
            body_chunk += body_header + json.dumps({'source': query_dict, 'params': var[i]}) + "\n"
        #print(body_chunk)
        result = self.elastic.msearch_template(index=index, body=body_chunk, max_concurrent_searches=50)
        
        list_response = []
        for i in range(nb_params):
            response = []
            for hit in result['responses'][i]['hits']['hits']:
                response.append(hit)
            list_response.append(response)
        return list_response
    
    def result_to_dataframe(self, res):
        list_df = []
        for item in res:
            flat_res = [flatten(x) for x in item]
            df = pd.DataFrame(flat_res)
            list_df.append(df)
        return list_df
    
    
    def export_data_to_es(self, settings, data, index: str, create_index=True, fail_if_index_exists=False):
        """
        Exporte les données dans elastic search
        
        :param settings: json du mapping ES
        :param data: data à exporter
        :param index: nom de l'index
        :param create_index: bool, doit_on créer l'index s'il n'existe pas ?
        :param fail_if_index_exists: if True, fails if index already exists
        """
        
        index_exists = self.elastic.indices.exists(index)
        if index_exists and fail_if_index_exists:
            print("Index ayant le même nom trouvé")
            raise RuntimeError('Index ayant le même nom trouvé')
#             self.elastic.indices.delete(index=index)
        else:
            if not index_exists and create_index:
                self.elastic.indices.create(index=index, ignore=400, body=settings)
                print("Index crée")

        deque(parallel_bulk(client = self.elastic
                            , index = index
                            , actions = gen_data_from_file(data, index)
                            , chunk_size = 2000
                            , thread_count = 4
                            , queue_size = 4)
              , maxlen = 0)
        
class async_ElasticDriver():
    
    def __init__(self, host, port, http_proxy=None, requests_json_dir='elastic'):
        """
        Classe de driver ES

        :param host: adresse du cluster (str ou liste de str)
        :param port: port à utiliser
        :param http_proxy: non utilisé (si proxy, le passer dans les variables d'env http_proxy et https_proxy)
        :param requests_json_dir: dossier contenant les définitions des requêtes en json
        """
    
        self.HOST: str = host
        self.port: str = port
            
        self.elastic = AsyncElasticsearch(self.HOST, maxsize=64, http_compress=True, timeout=120)
        self.requests_json_dir = requests_json_dir
        self.query_names = query_names
        self.query_dict = self.load_queries()
        self.loop=self.get_or_create_eventloop()
        
    def load_query(self, name: str):
        """
        Chargement d'un fichier json 

        :param name: nom du fichier.json
        :returns: requete en json
        """
        with open(name) as json_file:
            query = json.load(json_file)
        return query

    def load_queries(self):
        """
        Chargement de toutes les requêtes ES à passer 

        :returns: dict de requetes en json avec comme clé le nom de la requête
        """
        query_dict={}
        for query in self.query_names:
            query_dict[query]=self.load_query(os.path.join(self.requests_json_dir,query+'.json'))
        return query_dict   
    
    async def async_request(self, index: str, var: list ):
        """
        Execute un flot de requêtes ES
        
        :param index: nom de la table
        :param var: list des 'params'
        :returns: list, list de hits
        """
        nb_params: int = len(var)
        tasks = []
     
        for query in self.query_names:
            body_chunk=self.make_body(name_query=self.query_dict[query],index=index,var=var)
            #print(body_chunk)
            task = self.loop.create_task(
                        self.elastic.msearch_template(index=index, body=body_chunk, max_concurrent_searches=50)
                )
            tasks.append(task)

        return await asyncio.gather(*tasks)
    
    def make_body(self, name_query, index, var):
        """
        Met en forme le corps des requêtes ES
        
        :param name_query: nom de le requete à executer
        :param index: nom de la table
        :param var: list des 'params'
        :returns: body_chunk 
        """
        nb_params: int = len(var)
        body_header = json.dumps({"index" : index})+ "\n"
        body_chunk = ""
            
        for i in range(nb_params):
            body_chunk += body_header + json.dumps({'source': name_query, 'params': var[i]}) + "\n"
        return body_chunk
    
    def result_to_dataframe(self, res):
        list_df = []
        for item in res:
            flat_res = [flatten(x) for x in item]
            df = pd.DataFrame(flat_res)
            list_df.append(df)
        return list_df
    
    
    def export_data_to_es(self, settings, data, index: str, create_index=True, fail_if_index_exists=False):
        """
        Exporte les données dans elastic search
        
        :param settings: json du mapping ES
        :param data: data à exporter
        :param index: nom de l'index
        :param create_index: bool, doit_on créer l'index s'il n'existe pas ?
        :param fail_if_index_exists: if True, fails if index already exists
        """
        
        index_exists = self.elastic.indices.exists(index)
        if index_exists and fail_if_index_exists:
            print("Index ayant le même nom trouvé")
            raise RuntimeError('Index ayant le même nom trouvé')
#             self.elastic.indices.delete(index=index)
        else:
            if not index_exists and create_index:
                self.elastic.indices.create(index=index, ignore=400, body=settings)
                print("Index crée")

        deque(parallel_bulk(client = self.elastic
                            , index = index
                            , actions = gen_data_from_file(data, index)
                            , chunk_size = 2000
                            , thread_count = 4
                            , queue_size = 4)
              , maxlen = 0)
        
    def get_or_create_eventloop(self):
        try:
            return asyncio.get_event_loop()
        except RuntimeError as ex:
            if "There is no current event loop in thread" in str(ex):
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                return asyncio.get_event_loop()

def flatten(d, parent_key='', sep='_'):
    """
    Flatten un dictionnaire
    
    { a: { a: a , b: a}}
    =>
    {a_a :a, a_b: a}
    
    :param d: dict
    :param parent_k: char
    :param sep: char de delimitation
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

if __name__ == "__main__":
    
    elastic_host = ["http://es01-elastic-ssplab.marathon-ouest.containerip.dcos.thisdcos.directory:9200","http://es02-elastic-ssplab.marathon-ouest.containerip.dcos.thisdcos.directory:9200","http://es03-elastic-ssplab.marathon-ouest.containerip.dcos.thisdcos.directory:9200"]
    elastic_port = 9200
    elastic_index_bi = 'rp_2020_e'
    elastic_index_sirus = 'sirus_2020_e'
    
    elastic_driver = async_ElasticDriver(host=elastic_host, 
                                           port=elastic_port,
                                           requests_json_dir=os.path.realpath('elastic'))
    
    print(elastic_driver.query_names)
    print(elastic_driver.query_dict)