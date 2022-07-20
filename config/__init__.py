import yaml
import s3fs
import os

## TO DO:
# supprimer mot de passe minio
secret=""
### CONFIG GENERIQUE : POSTGRE, MINIO, ELASTIC

# 1. MINIO
fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': 'http://minio.ouest.innovation.insee.eu'},
                       key = 'ssplab',
                       secret = secret)

fs_path_to_settings = "ssplab/aiee2/data/settings.yml"

with fs.open(fs_path_to_settings) as f:
    settings = yaml.safe_load(f)
    
# 2. ELASTIC SEARCH CONFIG
elastic_host = settings['elasticsearch']['host']
elastic_port = settings['elasticsearch']['port']

# 3. DB CONFIG
db_host = settings['postgre_sql']['host']
db_port = settings['postgre_sql']['port']
db_dbname = settings['postgre_sql']['dbname']
db_user = settings['postgre_sql']['user']
db_password = settings['postgre_sql']['password']

### DEFINITION DE L'ANNEE
recap_year = str(2020)

### REQUETES ELASTIC
requests_json_dir=os.path.realpath(os.path.join('pipeline_siret_bi', 'elastic'))

query_names = [  'rsplus_and_adr'
                    , 'rsplus_and_latlon'
                    , 'exactrs_and_adr'
                    , 'exactrs_and_latlon'
                    , 'fullShould1_gestionRS-accro'
                    , 'fullShouldGen'
                    , 'FullShouldEnseignement'
                    , 'GenAndGeo1'
                    , 'simpleGeo2_must-RS-geo-naf-fonction-score'
                    , 'Geo4all'
                      ]
bi_query_names = ['robust_bi_query']

#load queries


#load geocode dicts
