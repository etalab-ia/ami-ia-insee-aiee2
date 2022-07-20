import os
import json, yaml
import s3fs
import pandas as pd
from datetime import datetime

#########################
# config manipulation
#########################

def load_config(save_dir):
    """
    Chargement d'une config
    
    :param save_dir: dossier de sauvegarde
    :returns: dict
    """
    with open(os.path.join(save_dir, "config.yaml")) as f:
        config = yaml.safe_load(f)
    return config


def save_config(config, save_dir):
    """
    Sauvegarde d'une config
    
    :param config: dict
    :param save_dir: dossier de sauvegarde
    :returns: None
    """
    with open(os.path.join(save_dir, "config.yaml"), 'w') as f:
        yaml.dump(config, f)
        

#########################
# Remote handling
#########################

def get_trainings_dir(config):
    """
    Récupération des chemins de sauvegarde globaux locaux et distants (où l'on stocke les dossiers de svg)
    
    :param config: dict
    :returns: str - chemin local, str - chemin distinct
    """
    local = os.path.join(config['local']['trainings_dir'], config['data']['nomenclature']['name'])
    remote = os.path.join(config['minio']['trainings_dir'], config['data']['nomenclature']['name'])
    return local, remote


def push_to_minio(save_dir):
    """
    Push du dossier sur minio
    
    :param save_dir: dossier à pusher (doit contenir un fichier paths.json)
    """
    config = load_config(save_dir)
    with open(os.path.join(save_dir, "paths.json")) as f:
        remote_training_dir = json.load(f)['remote_path']
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': config['minio']['endpoint']})
    fs.put(save_dir, remote_training_dir, recursive=True)


def pull_from_minio(endpoint, remote_training_dir, local_trainings_dir):
    """
    Pull d'un dossier de sauvegarde distant
    
    :param endpoint: adresse du endpoint minio
    :param remote_training_dir: chemin vers le training à récupérer
    :param local_trainings_dir: dossier des trainings global local
    """
    tokens = remote_training_dir.split('/')
    training_date = tokens[-2]
    training_ind = tokens[-1]
    local_training_dir = os.path.join(local_trainings_dir, f'{training_date}_{training_ind}')
    if os.path.exists(local_training_dir):
        raise ValueError(f'{local_training_dir} already exists')
    os.makedirs(local_training_dir, exist_ok=True)
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': endpoint})
    fs.get(remote_training_dir, local_training_dir, recursive=True)
    

#########################
# Helpers to get last trainings and best models
#########################

def get_last_local_training_dir(trainings_dir):
    """
    Renvoie le chemin vers le training local le plus récent
    
    :param trainings_dir: dossier de sauvegarde des trainings local
    :returns: str - path
    """
    latest_date = datetime.strptime('1990-01-01', "%Y-%m-%d").date()
    latest_ind = 0
    for td in os.listdir(trainings_dir):
        if 'data' in td or '.' in td:
            continue
        t_date, t_ind = td.split('_')
        t_date = datetime.strptime(t_date, "%Y-%m-%d").date()
        t_ind = int(t_ind)
        if t_date == latest_date and t_ind > latest_ind:
            latest_ind = t_ind
        if t_date > latest_date:
            latest_date = t_date
            latest_ind = t_ind
    return os.path.join(trainings_dir, latest_date.isoformat() + f'_{latest_ind}')


def get_best_savedmodel(training_dir):
    """
    renvoie le chemin vers le meilleur set de poids entraînés dans training_dir
    
    :param training_dir: str - chemin vers le training
    :returns: str - path
    """
    best_score = 10
    best_model = ""
    for td in os.listdir(os.path.join(training_dir, "train_weights")):
        score = float(td.split('-')[-1])
        if not best_model:
            best_model = td
            best_score = score
        if score < best_score:
            best_model = td
            best_score = score
    return best_model


##########################
# config analysis
##########################

def extract_config(remote_dir, fs):
    """
    Récupère la config d'un dossier distant de training
    
    :param remote_dir: chemin du training sur minio
    :param fs: s3fs file system
    :returns: dict
    """
    with fs.open(os.path.join(remote_dir, "config.yaml")) as f:
        config = yaml.load(f)
        config_extract = {}
        if "ngrams" in config and config['ngrams']['use']:
            config_extract['type'] = "trigrams" if config['ngrams']['ngrams_size'] == 3 else "bigrams"
        else:
            config_extract['type'] = "words"

        if 'fasttext' in config['data'] and config['data']['fasttext']['use']:
            config_extract['fasttext'] = str(config['data']['fasttext']['embedding_size'])
            if config['data']['fasttext'].get('trainable', False):
                config_extract['fasttext'] += "_trainable"
        else:
            config_extract['fasttext'] = "False"

        if isinstance(config['trainings'], list):
            config['trainings'] = config['trainings'][0]
        config_extract['model'] = config['trainings']['model']
        config_extract['inputs'] = config['trainings']['data']['input_columns']
        config_extract['embeddings_size'] = config['trainings']['model_params']['embedding_size']
        if 'fasttext' in config and config['fasttext']['use']:
            config_extract['embeddings_size'] = config['fasttext']['embedding_size']
        config_extract['blocks'] = config['trainings']['model_params']['nb_blocks']
        config_extract['heads'] = config['trainings']['model_params']['nb_heads']
        config_extract['ff_dim'] = config['trainings']['model_params']['ff_dim']
        config_extract['dense_sizes'] = config['trainings']['model_params']['dense_sizes']
        return config_extract
    
    
def extract_results(remote_dir, fs):
    """
    Récupère les résultats d'un dossier distant de training
    
    :param remote_dir: chemin du training sur minio
    :param fs: s3fs file system
    :returns: dict
    """
    results = {}
    results['train_scores'] = []
    if fs.exists(os.path.join(remote_dir, 'train_weights')):
        for f in fs.ls(os.path.join(remote_dir, 'train_weights')):
            results['train_scores'].append(os.path.basename(f).split('_')[-1])

    results['top_k']={}
    if fs.exists(os.path.join(remote_dir, 'top_k.json')):
        with fs.open(os.path.join(remote_dir, 'top_k.json')) as f:
            topk = json.load(f)
            results['top_k']= topk['top_k_perc']

    results['top_k_optim']={}
    if fs.exists(os.path.join(remote_dir, 'optim_top_k.json')):
        with fs.open(os.path.join(remote_dir, 'optim_top_k.json')) as f:
            topk = json.load(f)
            results['top_k_optim']= topk['top_k_perc']
            
    return results


def get_all_remote_results(remote_training_dir, fs):
    """
    Récupère les configs et results de tous les dossiers de training distants
    
    :param remote_training_dir: chemin de svg des trainings sur minio
    :param fs: s3fs file system
    :returns: df - configs, df - results
    """
    configs = []
    results = []
    for daily_dir in fs.ls(remote_training_dir):
        if 'data' in os.path.basename(daily_dir):
            continue
        for single_dir in fs.ls(daily_dir):
            try:
                config_extract = extract_config(single_dir, fs)
                config_extract['training_dir'] = "_".join(single_dir.split('/')[-2:])
                configs.append(config_extract)

                training_result = extract_results(single_dir, fs)
                training_result['training_dir'] = "_".join(single_dir.split('/')[-2:])
                results.append(training_result)
            except FileNotFoundError:
                continue

    configs = pd.DataFrame.from_dict(configs).set_index('training_dir')
    results = pd.DataFrame.from_dict(results).set_index('training_dir')
    for res in ['top_k', 'top_k_optim']:
        for v in [1, 3, 5, 10]:
            results[res.replace('k', str(v))] = results[res].map(lambda x: x.get(str(v), None))

    return configs, results



if __name__ == "__main__":
    
    push_to_minio("trainings/PCS/2020-12-07_1")