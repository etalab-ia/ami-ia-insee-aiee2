import os
import shutil
import tempfile
import logging
import s3fs
import glob

from typing import List, Optional
from pydantic import BaseModel, Field
import pandas as pd

from config import *
from data_import.bdd import PostGre_SQL_DB
from nomenclatures.script_run_top_k import load_model_from_save_dir, run_top_k_on_test
from nomenclatures.training_utils import load_config
from pipeline_siret_bi.elastic import ElasticDriver, async_ElasticDriver
from pipeline_siret_bi.training_classes.utils import delete_all_files_from_directory, move_all_files_from_directory_to_another
from pipeline_siret_bi.script_runtime import load_pipeline_and_model as load_siret, project as project_siret, geocode_new_bi
from pipeline_siret_bi.script_runtime import generate_echos, get_top_k, codage_automatique
from pipeline_bi_noncodable.script_runtime import load_pipeline_and_model as load_noncodables, project as project_noncodable


##################
#  Drivers
##################

HTTP_PROXY = os.environ.get('http_proxy', None)

def get_fs(endpoint):
    """
    Création d'un filesystem s3fs vers :param endpoint: avec http proxy si nécessaire
    """
    s3_config_kwargs={}
    secret=""
    if HTTP_PROXY:
        s3_config_kwargs={'proxies': {'https': HTTP_PROXY}}
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': endpoint}, config_kwargs=s3_config_kwargs)
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': 'http://minio.ouest.innovation.insee.eu'},
                       key = 'ssplab',
                       secret = secret)
    return fs


def get_bdd(fs_endpoint, minio_path_to_settings):
    """
    Création du driver BDD avec un s3fs existant
    """
    #return PostGre_SQL_DB(fs=get_fs(fs_endpoint), fs_path_to_settings=minio_path_to_settings)
    return PostGre_SQL_DB(host=db_host, port=db_port, dbname=db_dbname, user=db_user, password=db_password)


def get_elastic(config):
    """
    Création du driver ES
    """
    return ElasticDriver(host=elastic_host, 
                         port=elastic_port,
                         requests_json_dir=os.path.realpath(os.path.join('pipeline_siret_bi', 'elastic')))

def async_get_elastic(config):
    """
    Création du driver ES async
    """
    return async_ElasticDriver(host=elastic_host, 
                         port=elastic_port,
                         requests_json_dir=os.path.realpath(os.path.join('pipeline_siret_bi', 'elastic')))


def download_model(model_name, minio_endpoint, remote_dir, local_dir):
    """
    Fonction qui tente de récupérer un modèle depuis minio.

    :param model_name: nom du modèle (pour logging)
    :param minio_endpoint: url de minio
    :param remote_dir: dossier distant à télécharger
    :param local_dir: dossier local où télécharger
    :raises: Exception de s3fs en cas d'erreur
    """
    try:
        logging.info(f"Start downloading {model_name.upper()} model from {minio_endpoint} - {remote_dir}")
        os.makedirs(local_dir, exist_ok=True)
        fs = get_fs(minio_endpoint)
        fs.get(remote_dir, local_dir, recursive=True)
        logging.info('Download successful')
    except Exception as e:
        logging.error(f'Could not get {model_name.upper()} model from minio : {e.__class__.__name__} - {e}')
        logging.error(f"Arguments : {minio_endpoint} - {remote_dir}")
        raise e
    
##############
# Interfaces
##############

class BIModelInput(BaseModel):
    """
    Document model for BIs
    """
    cabbi: str = Field(None, title="cabbi du BI", max_length=20)
    rs_x: str = Field(None, title="nom déclaratif de l'entreprise", max_length=300)
    actet_x: str = Field(None, title="activité déclarative de l'entreprise", max_length=300)
    profs_x: Optional[str] = Field(None, title="profession salarié déclarée", max_length=300)
    profi_x: Optional[str] = Field(None, title="profession indépendante déclarée", max_length=300)
    profa_x: Optional[str] = Field(None, title="ancienne profession déclarée", max_length=300)
    numvoi_x: str = Field('', title="Numéro de l'adresse déclarée du lieu de travail", max_length=15)
    bister_x: Optional[str] = Field('', title="Indice de repetition", max_length=30)
    typevoi_x: Optional[str] = Field('', title="Type de voie de l'adresse déclarée du lieu de travail", max_length=100)
    nomvoi_x: Optional[str] = Field('', title="Nom de voie de l'adresse déclarée du lieu de travail", max_length=100)
    cpladr_x: Optional[str] = Field('', title="Complement d'adresse'", max_length=100)
    clt_x: Optional[str] = Field('', title="commune déclarée du lieu de travail", max_length=100)
    dlt_x: Optional[str] = Field('', title="département du lieu de travail", max_length=100)
    plt_x: Optional[str] = Field('', title="Pays du lieu de travail", max_length=100)
    clt_c_c: str = Field(None, title="Code departement-commune lieu de travail", max_length=10)
    vardompart_x: Optional[str] = Field('', title="Travaille a un endroit variable ou a domicile", max_length=10)

class BIModelRequest(BaseModel):
    """
    Request model for BIs (list of BIModelInput)
    """
    documents: List[BIModelInput]


class BISaveInput(BaseModel):
    """
    Document model for saving BI to data
    """
    cabbi: str = Field(None, title="cabbi du BI", max_length=20)

class BISaveRequest(BaseModel):
    """
    Request model for BIs (list of BISaveInput)
    """
    documents: List[BISaveInput]

############
# Models
############

def load_nomenclature_model(config, nomenclature_name):
    """
    Load model. Get it from minio if necessary. Tries twice if error the first time

    :param config: app config (dict)
    :param nomenclature_name: str. naf or pcs
    :returns: 
        model_config, data_cleaner, data_formatter, model, top1_model
    """
    model_remote_path = config['models'][f'{nomenclature_name}_model']
    local_model = os.path.join(config['models']['local_directory'], nomenclature_name,
                                os.path.basename(model_remote_path))
    local_model = os.path.abspath(local_model)
    nb_retries = 1
    load_try = 0
    while load_try <= nb_retries:      
        try:
            if not os.path.exists(local_model):
                download_model(nomenclature_name, 
                               config['models']['minio_endpoint'], 
                               model_remote_path, 
                               local_model)
            try:
                logging.info(f"Start loading {nomenclature_name.upper()} model")
                model_config = load_config(local_model)
                model_config['trainings']['data']['gt_column'] = None  # for prediction
                bdd = get_bdd(config['models']['minio_endpoint'],
                              config['bdd']['minio_path_to_settings_file'])
                nomenclature, data_cleaner, data_formatter, model, top1_model = \
                    load_model_from_save_dir(local_model, bdd=bdd)
                logging.info(f"{nomenclature_name.upper()} model loaded")
                return model_config, data_cleaner, data_formatter, model, top1_model
            except Exception as e:
                logging.error(f'Could not load {nomenclature_name.upper()} model : {e.__class__.__name__} - {e}')
                raise e

        except:
            if load_try < nb_retries:
                logging.error(f'Cleaning and retrying to load {nomenclature_name.upper()} model')
                if os.path.exists(local_model):
                    shutil.rmtree(local_model)
            load_try += 1
            continue

    logging.error(f'Could not load {nomenclature_name.upper()} model. Exiting...')
    raise RuntimeError()


def load_siret_model(config):
    """
    Load Siret model. get it from minio if needed. tries twice if fails the first time

    :param config: app config (dict)
    :returns: model_config, model_cleaners, model_processes, model, meta_model, threshold
    """
    model_remote_path = config['models'][f'siret_model']
    local_model = os.path.join(config['models']['local_directory'], 'siret',
                                os.path.basename(model_remote_path))
    local_model = os.path.abspath(local_model)
    nb_retries = 1
    load_try = 0
    while load_try <= nb_retries:      
        try:
            if not os.path.exists(local_model):
                download_model('siret', 
                               config['models']['minio_endpoint'], 
                               model_remote_path, 
                               local_model)
            try:
                logging.info(f"Start loading SIRET model")
                fs = get_fs(config['models']['minio_endpoint'])
                model_config, model_cleaners, model_processes, model, meta_model, threshold = \
                    load_siret(os.path.join(local_model, "config.yaml"), fs=fs)
                logging.info(f"SIRET model loaded")
                return model_config, model_cleaners, model_processes, model, meta_model, threshold
            except Exception as e:
                logging.error(f'Could not load SIRET model : {e.__class__.__name__} - {e}')
                raise e

        except:
            if load_try < nb_retries:
                logging.error(f'Cleaning and retrying to load SIRET model')
                if os.path.exists(local_model):
                    shutil.rmtree(local_model)
            load_try += 1
            continue

    logging.error(f'Could not load SIRET model. Exiting...')
    raise RuntimeError()


def load_noncodables_model(config):
    """
    Load non-codables model. get it from minio if needed. tries twice if fails the first time

    :param config: app config (dict)
    :returns: model_config, model_cleaners, model_processes, model
    """
    model_remote_path = config['models'][f'noncodables_model']
    local_model = os.path.join(config['models']['local_directory'], 'noncodables',
                                os.path.basename(model_remote_path))
    local_model = os.path.abspath(local_model)
    nb_retries = 1
    load_try = 0
    while load_try <= nb_retries:      
        try:
            if not os.path.exists(local_model):
                download_model('noncodables', 
                               config['models']['minio_endpoint'], 
                               model_remote_path, 
                               local_model)
            try:
                logging.info(f"Start loading NON_CODABLES model")
                fs = get_fs(config['models']['minio_endpoint'])
                model_config, model_cleaners, model_processes, model = \
                    load_noncodables(os.path.join(local_model, "config.yaml"), fs=fs)
                model.load_model(os.path.join(local_model, 'model', local_model, 'model.h5'),
                                 os.path.join(local_model, 'model', local_model, 'dict_info.p'))
                logging.info(f"NON_CODABLES model loaded")
                return model_config, model_cleaners, model_processes, model
            except Exception as e:
                logging.error(f'Could not load NON_CODABLES model : {e.__class__.__name__} - {e}')
                raise e

        except:
            if load_try < nb_retries:
                logging.error(f'Cleaning and retrying to load NON_CODABLES model')
                if os.path.exists(local_model):
                    shutil.rmtree(local_model)
            load_try += 1
            continue

    logging.error(f'Could not load SIRET model. Exiting...')
    raise RuntimeError()


class Models:

    def __init__(self, app_config):
        """
        Classe chargeant les modèles et permettant la prédiction
        """
        self.app_config = app_config
        self.naf_model_elements = None
        self.pcs_model_elements = None
        self.siret_model_elements = None
        self.noncodables_model_elements = None

    def load_all_models(self):
        self.naf_model_elements = load_nomenclature_model(self.app_config, "naf")
        self.pcs_model_elements = load_nomenclature_model(self.app_config, "pcs")
        self.siret_model_elements = load_siret_model(self.app_config)
        self.noncodables_model_elements = load_noncodables_model(self.app_config)

    @staticmethod
    def predict_nomenclature(model_config, data_cleaner, data_formatter, model, top1_model,
                             request_data, main_field=""):
        """
        Predict nomenclature top-10 for request data and given model

        :param model_config, data_cleaner, data_formatter, model, top1_model: loaded model
        :param request_data: BIModelRequest
        :param main_field: champs input principal de la nomenclature (pour message d'erreur)
        :return: [dict] avec:
            dict = {'cabbi': cabbi, 'status': 'ok', 'predictions': [[code_0, score_0], ...]} ou
            dict = {'cabbi': cabbi, 'status': 'error', 'error': 'message d'erreur'}
        """
        input_df = pd.DataFrame([d.dict() for d in request_data.documents])
        results = []
        results_cabbis = []
        try:
            _, result_df = run_top_k_on_test(model_config, [input_df], len(input_df), 
                                            data_cleaner, data_formatter, model)
            for _, values in result_df.iterrows():
                result = {}
                result['cabbi'] = str(values['cabbi'])
                results_cabbis.append(str(values['cabbi']))
                result['predictions'] = list(zip(values['top_k_codes'], values['top_k_similarities']))
                result['codage_auto'] = False
                result['codage_auto_proba'] = 0
                result['status'] = 'ok'
                results.append(result)
        except RuntimeError as e:
            if str(e) == 'No test doc retrieved from BDD':
                # all input documents were missing the main field
                pass
            else:
                raise e

        for cabbi in [d.cabbi for d in request_data.documents]:
            if cabbi not in results_cabbis:
                # documents that have not been processed due to missing value
                result = {
                    'cabbi': cabbi,
                    'status': 'error',
                    'error': str(ValueError(f'{main_field} is missing'))
                }
                results.append(result)

        return results

    def predict_naf(self, request_data: BIModelRequest):
        """
        Predict NAF top-10 for request data and given model

        :param request_data: BIModelRequest
        :return: [dict] avec:
            dict = {'cabbi': cabbi, 'status': 'ok', 'predictions': [[code_0, score_0], ...], 'codage_auto': bool, 'codage_auto_score': float} ou
            dict = {'cabbi': cabbi, 'status': 'error', 'error': 'message d'erreur'}
        """
        return self.predict_nomenclature(*self.naf_model_elements, request_data,
                                         main_field="actet_x")

    def predict_pcs(self, request_data: BIModelRequest):
        """
        Predict PCS top-10 for request data and given model

        :param request_data: BIModelRequest
        :return: [dict] avec:
            dict = {'cabbi': cabbi, 'status': 'ok', 'predictions': [[code_0, score_0], ...], 'codage_auto': bool, 'codage_auto_score': float} ou
            dict = {'cabbi': cabbi, 'status': 'error', 'error': 'message d'erreur'}
        """
        return self.predict_nomenclature(*self.pcs_model_elements, request_data,
                                         main_field="prof(s/i/a)_x")

    def predict_siret(self, request_data: BIModelRequest):
        """
        Predict SIRET top-10 for request data and given model

        :param request_data: BIModelRequest
        :return: [dict] avec:
            dict = {'cabbi': cabbi, 'status': 'ok', 'predictions': [[code_0, score_0], ...], 'codage_auto': bool, 'codage_auto_score': float} ou
            dict = {'cabbi': cabbi, 'status': 'error', 'error': 'message d'erreur'}
        """
        model_config, model_cleaners, model_processes, model, meta_model, threshold = self.siret_model_elements

        #Récupération des liens externes
        bdd = get_bdd(self.app_config['models']['minio_endpoint'],
                      self.app_config['bdd']['minio_path_to_settings_file'])
        bdd_sirus_table = self.app_config['bdd']['sirus_table']
        bdd_sirus_proj_table = self.app_config['bdd']['sirus_proj_table']
        bdd_naf_proj_table = self.app_config['bdd']['naf_proj_table']

        elastic = get_elastic(self.app_config)
        async_elastic = async_get_elastic(self.app_config)
        elastic_bi_index = self.app_config['elasticsearch']['index_bi']
        elastic_sirus_index = self.app_config['elasticsearch']['index_sirus']

        addok_urls = self.app_config['addok_apis']

        # dossiers locaux
        #work_dir = os.path.abspath('runtime_siret')
        #on crée un répertoire temporaire "aléatoire" par requête
        tempdir = tempfile.TemporaryDirectory()
        work_dir = tempdir.name
        BI_EMB_DIRECTORY = os.path.join(work_dir, "emb_bi")
        
        # nettoyage
        #if os.path.exists(work_dir):
        #    shutil.rmtree(work_dir)
        os.makedirs(BI_EMB_DIRECTORY, exist_ok=True)

        #projection du BI
        logging.info('Projection BI...')
        input_df = pd.DataFrame([d.dict() for d in request_data.documents])

        for col in input_df.columns:
            input_df[col] =  input_df[col].apply(str)
        input_df = input_df.fillna('')
        input_df['siretc'] = ''
        project_siret(input_df, 
                      model_config, model_cleaners, model_processes, model, 
                      projections_dir=BI_EMB_DIRECTORY, work_dir=os.path.join(work_dir, 'tmp'))
        logging.info("Projection terminé")
        
        # get echos
        def predict_unknown_naf(bis_for_naf):
            cabbis = bis_for_naf['cabbi'].values.tolist()
            naf_request = BIModelRequest(documents=[d for d in request_data.documents if d.cabbi in cabbis])
            proj_nafs = self.predict_naf(naf_request)
            # reformat results
            for i, result in enumerate(proj_nafs):
                if 'predictions' not in result:
                    # doc was missing actet_x
                    result['predictions'] = [['', 0] for i in range(10)]
                    logging.warning(f'no NAF prediction for {result["cabbi"]}: {result["error"]}')
                    del proj_nafs[i]['error']
                for k in range(10):
                    proj_nafs[i][f'naf_code_{k}'] = result['predictions'][k][0]
                    proj_nafs[i][f'naf_score_{k}'] = result['predictions'][k][1]
                del proj_nafs[i]['predictions']
                del proj_nafs[i]['status']
            proj_nafs = pd.DataFrame.from_dict(proj_nafs)
            proj_nafs = proj_nafs.set_index("cabbi")
            return proj_nafs

        input_df, echos = generate_echos(input_df,
                                         bdd, bdd_naf_proj_table, predict_unknown_naf,
                                         elastic, async_elastic, elastic_bi_index, elastic_sirus_index,
                                         addok_urls=addok_urls)

        # calcul du top-k
        logging.info('topk...')
        topk = get_top_k(echos, 
                         bi_directory_path=BI_EMB_DIRECTORY, 
                         naf_predictions_df=input_df,
                         coeff_naf_score=model_config['final_score_coeffs']['coeff_naf_score'],
                         coeff_nb_fois_trouve=model_config['final_score_coeffs']['coeff_nb_fois_trouve'],
                         driver_bdd=bdd,
                         sirus_bdd_table=bdd_sirus_table,
                         sirus_proj_bdd_table=bdd_sirus_proj_table,
                         naf_proj_bdd_table=bdd_naf_proj_table)
        logging.info('topk terminé')

        #calcul du codage auto (le top-1 est-il suffisamment bon ?)
        logging.info('codage automatique...')
        topk = codage_automatique(topk, meta_model, threshold)
        logging.info('codage automatique terminé')

        #formattage des résultats
        results = []
        for key, values in topk.items():
            result = {}
            result['cabbi'] = key
            result['status'] = 'ok'
            result['predictions'] = [(v['siret'], v['similarite_final']) for v in values]
            result['codage_auto'] =bool(values[0]['codage_auto'].item())
            result['codage_auto_proba'] = values[0]['codage_auto_proba'].item()
            results.append(result)
        #on détruit le répertoire temp de l'utilisateur 
        tempdir.cleanup()
        return results

    def predict_noncodable(self, request_data: BIModelRequest):
        """
        Prédiction des non-codables

        :param request_data: BIModelRequest
        :return: [dict] avec
            dict = {'cabbi': cabbi, 'status': 'ok', 'non_codable': bool, 'predictions': [score_0, score_1]}
        """
        model_config, model_cleaners, model_processes, model = self.noncodables_model_elements

        work_dir = os.path.abspath('non_codable_runtime')
        model_dir = os.path.join(self.app_config['models']['local_directory'], 'noncodables',
                                os.path.basename(self.app_config['models'][f'noncodables_model']))
        model_dir = os.path.abspath(model_dir)


        logging.info('Classification BI...')
        input_df = pd.DataFrame([d.dict() for d in request_data.documents])
        classif = project_noncodable(input_df, model_config, model_cleaners, model_processes, model, 
                                    work_dir=work_dir, model_dir=model_dir)
        logging.info("Classification terminé")

        results = []
        for key, decision, scores in zip([d.cabbi for d in request_data.documents], classif[0], classif[1]):
            result = {}
            result['cabbi'] = key
            result['status'] = 'ok'
            result['non_codable'] = decision.item()
            result['predictions'] = scores.tolist()
            results.append(result)

        return results

    def push_bi_to_datasources(self, request_data: BISaveRequest):
        results = []
        cabbi = None
        for doc in request_data.documents:
            try:
                cabbi = doc.cabbi
                # get BI data from postgres
                #Récupération des liens externes
                bdd = get_bdd(self.app_config['models']['minio_endpoint'],
                                self.app_config['bdd']['minio_path_to_settings_file'])
                bdd_rp_table = self.app_config['bdd']['bi_table']
                bdd_rp_geo_table = self.app_config['bdd']['bi_geo_table']
                cabbi_df = bdd.read_from_sql(f"SELECT * FROM {bdd_rp_table} WHERE cabbi = '{cabbi}'")
                
                #cabbi_df = bdd.read_from_sql(f"""SELECT * FROM {bdd_rp_table} AS t1 
                #LEFT OUTER JOIN {bdd_rp_geo_table} AS t2 
                #ON t1.cabbi = t2.cabbi
                #WHERE t1.cabbi = '{cabbi}'""")
                
                # calculate geocoding
                addok_urls = self.app_config['addok_apis']
                logging.info("Appel de l'api de geo-encodage")
                for col in ['latitude', 'longitude', 'geo_adresse']:
                    if col in cabbi_df.columns:
                        cabbi_df = cabbi_df.drop([col], axis=1)
                cabbi_df = geocode_new_bi(cabbi_df, addok_urls)
                cabbi_df["latitude"] = cabbi_df["latitude"].astype(str)
                cabbi_df["longitude"] = cabbi_df["longitude"].astype(str)
                
                # push vers elastic
                elastic = get_elastic(self.app_config)
                elastic_bi_index = self.app_config['elasticsearch']['index_bi']
                
                cabbi_df['location'] = cabbi_df['latitude'] +", " + cabbi_df['longitude']
                cabbi_df = cabbi_df.fillna('')
                settings = elastic.load_query("settings_refined_bi")
                # insertion et pas update car l'index ES ne contient que des BI validés (donc pas encore celui-ci)
                elastic.export_data_to_es(settings, cabbi_df, elastic_bi_index, create_index=False)

                results.append({
                    'cabbi': cabbi,
                    'status': 'ok'
                })
            except Exception as e:
                logging.error( f"push_bi_to_datasources - CABBI {cabbi} - {e.__class__.__name__} : {str(e)}")
                results.append({
                    'cabbi': cabbi,
                    'status': 'error',
                    'error': f"push_bi_to_datasources - CABBI {cabbi} - {e.__class__.__name__} : {str(e)}"
                })
            # done
            # pas besoin de calculer la projection, puisqu'on prend celle du SIRET associé
            # update table des prédictions naf => Pas besoin, elles ne sont appelées que pour les BI non validés