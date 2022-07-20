"""
Fichier des fonctions utilitaires


Author : bsanchez@starclay.fr
date : 06/08/2020
"""

import os 
import s3fs
import yaml


def incrementTmpFile(df_path):
    """
    Les fichiers dans le dossier tmp sont identifié par un ID integer. A chaque appel 
    d'un transformer, l'id est incrementer de 1. Cette fonction se charge de retourner ce nouvel id.
    
    :param df_path: nom du fichier dans tmp
    
    :returns: new_name: le nouveau nom fichier incrementer de 1
    """
    try:
        base = os.path.basename(df_path)
        name = os.path.splitext(base)[0]
        try:
            new_name = int(name)
            new_name += 1
        except ValueError:
            new_name = 0
        new_name = str(new_name)
        return new_name
    except ValueError:
            print(F"Erreur d'incrementation id fichier tmp") 
            sys.exit(2)

def save_file_on_minio(file_name, dir_name):
    """
    Les fichiers dans le dossier tmp sont identifié par un ID integer. A chaque appel 
    d'un transformer, l'id est incrementer de 1. Cette fonction se charge de retourner ce nouvel id.
    
    Les variables suivantes doivent être dans l'environnement: 
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN
    
    :param df_path: nom du fichier dans tmp
    
    :returns: new_name: le nouveau nom fichier incrementer de 1
    """
        
    conf_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'config.yaml')
    
    try:
        with open(conf_file) as f:
            configs = yaml.safe_load(f)
    except ValueError:
            print(F"Erreur dans le chargement du fichier de configuration {conf_file}") 
            sys.exit(2)
    try:
        with open(file_name, mode='rb') as file: 
            fileContent = file.read()
    except ValueError:
            print(F"Erreur de lecture du fichier {file_name}") 
            sys.exit(2)
        
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': configs['minio']['endpoint']})
        
    try:
        with fs.open(f"{dir_name}/{file_name}", 'wb') as f:
            f.write(fileContent)
    except ValueError:
        print(F"Erreur d'écriture sur minio {file_name}") 
        sys.exit(2)
                
def load_file_on_minio(path, file_name):
    """
    Va chercher un fichier minio et le copie en local
    
    Les variables suivantes doivent être dans l'environnement: 
    AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_SESSION_TOKEN
    
    :param path: chemin minio du fichier à récupérer
    :param file_name: chemin local où copier le fichier distant
    :returns: None
    """
    conf_file = os.path.join(os.path.dirname(__file__), 'config.yaml')
    
    try:
        with open(conf_file) as f:
            configs = yaml.safe_load(f)
    except ValueError:
            print(F"Erreur dans le chargement du fichier de configuration {conf_file}") 
            sys.exit(2)
            
    fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': configs['minio']['endpoint']})
    try:
        fs.get(path,file_name)
    except ValueError:
        print(F"Impossible de récupérer {file_name} sur minio") 
        sys.exit(2)