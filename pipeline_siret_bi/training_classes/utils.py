"""
Fichier des fonctions utilitaires


Author : bsanchez@starclay.fr
date : 06/08/2020
"""

import os 
import s3fs
import yaml
import numba
from numba import jit
import shutil
import sys
import json
import numpy as np


def move_all_files_from_directory_to_another(source_dir: str, target_dir: str):
    """
    Deplace tous les fichiers d'un dossier à un autre
    params: source_dir
    params: target_dir
    
    :return:
        void
    """
    file_names = os.listdir(source_dir)
    for file_name in file_names:
        shutil.move(os.path.join(source_dir, file_name), target_dir)
        
def delete_all_files_execpt_from_directory(dir_name : str, files_to_keep: list):
    """
    Supprimes tous les fichier d'un dossier sauf certains
    Ne pas oublier le '/' 
    
    :params dir_name: nom dossier
    :params files_to_keep: list à garder
    
    :return:
        void
    """
    
    filelist = [f for f in os.listdir(dir_name)]
    for f in filelist:
        if os.path.basename(f) not in files_to_keep:
            os.remove(os.path.join(dir_name, f))

def delete_all_files_from_directory(dir_name : str):
    """
    Supprimes tous les fichier d'un dossier
    Ne pas oublier le '/' 
    
    :params source_dir: nom fichier
    :params target_dir: nom fichier
    
    :return:
        void
    """
    filelist = [f for f in os.listdir(dir_name)]
    for f in filelist:
        os.remove(os.path.join(dir_name, f))

def incrementTmpFile(df_path):
    """
    Les fichiers dans le dossier tmp sont identifié par un ID integer. A chaque appel 
    d'un transformer, l'id est incrementer de 1. Cette fonction se charge de retourner ce nouvel id.
    
    :params df_path: nom du fichier dans tmp
    
    :return new_name: le nouveau nom fichier incrementer de 1
    """
    try:
        base = os.path.basename(df_path)
        name = os.path.splitext(base)[0]
        new_name = int(name)
        new_name += 1
        new_name = str(new_name)
        return new_name
    except ValueError:
        print(F"Erreur d'incrementation id fichier tmp") 
        sys.exit(2)

def save_file_on_minio(file_name, dir_name):
    """
    Sauvegarde un fichier sur minio
    
    :params file_name: nom du fichier à save
    :params dir_name: chemin dans minio
    """
        
    conf_file = os.path.join(os.path.dirname(__file__), '../config.yaml')
    
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

def push_to_minio(save_dir, fs=None):
    """
    Push du dossier sur minio
    
    :param save_dir: dossier à pusher (doit contenir un fichier paths.json)
    :param fs: minio file system
    """
    config = load_config(save_dir)
    with open(os.path.join(save_dir, "paths.json")) as f:
        remote_training_dir = json.load(f)['remote_path']
    if not fs:
        fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': config['minio']['endpoint']})
    fs.put(save_dir, remote_training_dir, recursive=True)
    
def load_file_on_minio(path, file_name, fs=None):
    """
    Charger un fichier depuis minio
    
    :params file_name: nom du fichier à charger
    :params dir_name: chemin dans minio
    :param fs: minio file system
    """
    
    conf_file = os.path.join(os.path.dirname(__file__), 'config.yaml')
    
    try:
        with open(conf_file) as f:
            configs = yaml.safe_load(f)
    except ValueError:
            print(F"Erreur dans le chargement du fichier de configuration {conf_file}") 
            sys.exit(2)
    if not fs:        
        fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': configs['minio']['endpoint']})
    try:
        fs.get(path,file_name)
    except ValueError:
        print(F"Impossible de récupérer {file_name} sur minio") 
        sys.exit(2)
        
@jit(nopython=True, fastmath=True)
def cosine(u, v, w=None):
    """
    :purpose:
        Computes the cosine similarity between two 1D arrays
        Unlike scipy's cosine distance, this returns similarity, which is 1 - distance
    :params:
        u, v   : input arrays, both of shape (n,)
        w      : weights at each index of u and v. array of shape (n,)
                if no w is set, it is initialized as an array of ones
                such that it will have no impact on the output
    :returns:
        cosine  : float, the cosine similarity between u and v
    :example:
        >>> from fastdist import fastdist
        >>> import numpy as np
        >>> u, v, w = np.random.RandomState(seed=0).rand(10000, 3).T
        >>> fastdist.cosine(u, v, w)
        0.7495065944399267
    """
    n = len(u)
    w = init_w(w, n)
    num = 0
    u_norm, v_norm = 0, 0
    for i in range(n):
        num += u[i] * v[i] * w[i]
        u_norm += abs(u[i]) ** 2 * w[i]
        v_norm += abs(v[i]) ** 2 * w[i]

    denom = (u_norm * v_norm) ** (1 / 2)
    return num / denom


@jit(nopython=True, fastmath=True)
def cosine_vector_to_matrix(u, m):
    """
    :purpose:
        Computes the cosine similarity between a 1D array and rows of a matrix
    :params:
        u      : input vector of shape (n,)
        m      : input matrix of shape (m, n)
    :returns:
        cosine vector  : np.array, of shape (m,) vector containing cosine similarity between u
                        and the rows of m
    :example:
        >>> from fastdist import fastdist
        >>> import numpy as np
        >>> u = np.random.RandomState(seed=0).rand(10)
        >>> m = np.random.RandomState(seed=0).rand(100, 10)
        >>> fastdist.cosine_vector_to_matrix(u, m)
        (returns an array of shape (100,))
    """
    norm = 0
    for i in range(len(u)):
        norm += abs(u[i]) ** 2
    u = u / norm ** (1 / 2)
    for i in range(m.shape[0]):
        norm = 0
        for j in range(len(m[i])):
            norm += abs(m[i][j]) ** 2
        m[i] = m[i] / norm ** (1 / 2)
    return np.dot(u, m.T)


@jit(nopython=True, fastmath=True)
def cosine_matrix_to_matrix(a, b):
    """
    :purpose:
    Computes the cosine similarity between the rows of two matrices
    :params:
    a, b   : input matrices of shape (m, n) and (k, n)
             the matrices must share a common dimension at index 1
    :returns:
    cosine matrix  : np.array, an (m, k) array of the cosine similarity
                     between the rows of a and b
    :example:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> a = np.random.RandomState(seed=0).rand(10, 50)
    >>> b = np.random.RandomState(seed=0).rand(100, 50)
    >>> fastdist.cosine_matrix_to_matrix(a, b)
    (returns an array of shape (10, 100))
    """
    for i in range(a.shape[0]):
        norm = 0
        for j in range(len(a[i])):
            norm += abs(a[i][j]) ** 2
        a[i] = a[i] / norm ** (1 / 2)
    for i in range(b.shape[0]):
        norm = 0
        for j in range(len(b[i])):
            norm += abs(b[i][j]) ** 2
        b[i] = b[i] / norm ** (1 / 2)
    return np.dot(a, b.T)


@jit(nopython=True, fastmath=True)
def cosine_pairwise_distance(a, return_matrix=False):
    """
    :purpose:
    Computes the cosine similarity between the pairwise combinations of the rows of a matrix
    :params:
    a      : input matrix of shape (n, k)
    return_matrix : bool, whether to return the similarity as an (n, n) matrix
                    in which the (i, j) element is the cosine similarity
                    between rows i and j. if true, return the matrix.
                    if false, return a (n choose 2, 1) vector of the
                    similarities
    :returns:
    cosine matrix  : np.array, either an (n, n) matrix if return_matrix=True,
                     or an (n choose 2, 1) array if return_matrix=False
    :example:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> a = np.random.RandomState(seed=0).rand(10, 50)
    >>> fastdist.cosine_pairwise_distance(a, return_matrix=False)
    (returns an array of shape (45, 1))
    alternatively, with return_matrix=True:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> a = np.random.RandomState(seed=0).rand(10, 50)
    >>> fastdist.cosine_pairwise_distance(a, return_matrix=True)
    (returns an array of shape (10, 10))
    """
    n = a.shape[0]
    rows = np.arange(n)
    perm = [(rows[i], rows[j]) for i in range(n) for j in range(i + 1, n)]
    for i in range(n):
        norm = 0
        for j in range(len(a[i])):
            norm += abs(a[i][j]) ** 2
        a[i] = a[i] / norm ** (1 / 2)

    if return_matrix:
        out_mat = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                out_mat[i][j] = np.dot(a[i], a[j])
        return out_mat + out_mat.T
    else:
        out = np.zeros((len(perm), 1))
        for i in range(len(perm)):
            out[i] = np.dot(a[perm[i][0]], a[perm[i][1]])
        return out
    
@jit(nopython=True, fastmath=True)
def vector_to_matrix_distance(u, m, metric, metric_name):
    """
    :purpose:
    Computes the distance between a vector and the rows of a matrix using any given metric
    :params:
    u      : input vector of shape (n,)
    m      : input matrix of shape (m, n)
    metric : the function used to calculate the distance
    metric_name : str of the function name. this is only used for
                  the if statement because cosine similarity has its
                  own function
    distance vector  : np.array, of shape (m,) vector containing the distance between u
                       and the rows of m
    :example:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> u = np.random.RandomState(seed=0).rand(10)
    >>> m = np.random.RandomState(seed=0).rand(100, 10)
    >>> fastdist.vector_to_matrix_distance(u, m, fastdist.cosine, "cosine")
    (returns an array of shape (100,))
    :note:
    the cosine similarity uses its own function, cosine_vector_to_matrix.
    this is because normalizing the rows and then taking the dot product
    of the vector and matrix heavily optimizes the computation. the other similarity
    metrics do not have such an optimization, so we loop through them
    """

    if metric_name == "cosine":
        return cosine_vector_to_matrix(u, m)

    n = m.shape[0]
    out = np.zeros((n), dtype=np.float32)
    for i in range(n):
        out[i] = metric(u, m[i])
    return out


@jit(nopython=True, fastmath=True)
def matrix_to_matrix_distance(a, b, metric, metric_name):
    """
    :purpose:
    Computes the distance between the rows of two matrices using any given metric
    :params:
    a, b   : input matrices either of shape (m, n) and (k, n)
             the matrices must share a common dimension at index 1
    metric : the function used to calculate the distance
    metric_name : str of the function name. this is only used for
                  the if statement because cosine similarity has its
                  own function
    :returns:
    distance matrix  : np.array, an (m, k) array of the distance
                       between the rows of a and b
    :example:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> a = np.random.RandomState(seed=0).rand(10, 50)
    >>> b = np.random.RandomState(seed=0).rand(100, 50)
    >>> fastdist.matrix_to_matrix_distance(a, b, fastdist.cosine, "cosine")
    (returns an array of shape (10, 100))
    :note:
    the cosine similarity uses its own function, cosine_matrix_to_matrix.
    this is because normalizing the rows and then taking the dot product
    of the two matrices heavily optimizes the computation. the other similarity
    metrics do not have such an optimization, so we loop through them
    """
    if metric_name == "cosine":
        return cosine_matrix_to_matrix(a, b)
    n, m = a.shape[0], b.shape[0]
    out = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            out[i][j] = metric(a[i], b[j])
    return out


@jit(nopython=True, fastmath=True)
def matrix_pairwise_distance(a, metric, metric_name, return_matrix=False):
    """
    :purpose:
    Computes the distance between the pairwise combinations of the rows of a matrix
    :params:
    a      : input matrix of shape (n, k)
    metric : the function used to calculate the distance
    metric_name   : str of the function name. this is only used for
                    the if statement because cosine similarity has its
                    own function
    return_matrix : bool, whether to return the similarity as an (n, n) matrix
                    in which the (i, j) element is the cosine similarity
                    between rows i and j. if true, return the matrix.
                    if false, return a (n choose 2, 1) vector of the
                    similarities
    :returns:
    distance matrix  : np.array, either an (n, n) matrix if return_matrix=True,
                       or an (n choose 2, 1) array if return_matrix=False
    :example:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> a = np.random.RandomState(seed=0).rand(10, 50)
    >>> fastdist.matrix_pairwise_distance(a, fastdist.euclidean, "euclidean", return_matrix=False)
    (returns an array of shape (45, 1))
    alternatively, with return_matrix=True:
    >>> from fastdist import fastdist
    >>> import numpy as np
    >>> a = np.random.RandomState(seed=0).rand(10, 50)
    >>> fastdist.matrix_pairwise_distance(a, fastdist.euclidean, "euclidean", return_matrix=True)
    (returns an array of shape (10, 10))
    """
    if metric_name == "cosine":
        return cosine_pairwise_distance(a, return_matrix)

    else:
        n = a.shape[0]
        rows = np.arange(n)
        perm = [(rows[i], rows[j]) for i in range(n) for j in range(i + 1, n)]
        if return_matrix:
            out_mat = np.zeros((n, n))
            for i in range(n):
                for j in range(i):
                    out_mat[i][j] = metric(a[i], a[j])
            return out_mat + out_mat.T
        else:
            out = np.zeros((len(perm), 1))
            for i in range(len(perm)):
                out[i] = metric(a[perm[i][0]], a[perm[i][1]])
            return out