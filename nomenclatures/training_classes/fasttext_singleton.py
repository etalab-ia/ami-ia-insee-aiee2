import os
import shutil
import s3fs
import logging
import fasttext
import fasttext.util

class FasttextSingleton:
    class __fasttext:
        def __init__(self, local_path, embeddings_dim=300):
            if os.path.exists(os.path.join(local_path, f'cc.fr.{embeddings_dim}.bin')):
                self.model_file = os.path.join(local_path, f'cc.fr.{embeddings_dim}.bin')
            else:
                self.model_file = os.path.join(local_path, f'cc.fr.300.bin')
            logging.getLogger('FasttextSingleton').info(f"Loading FastText model {self.model_file}...")
            ft_model = fasttext.load_model(self.model_file)
            if embeddings_dim != 0 and embeddings_dim < ft_model.get_dimension():
                logging.getLogger('FasttextSingleton').info(f'Reducing FastText model to {embeddings_dim}...')
                fasttext.util.reduce_model(ft_model, int(embeddings_dim)) # Must be divisible by number of attention head
            self.ft = ft_model
            logging.getLogger('FasttextSingleton').info("Loaded FastText model")

    local_path=""
    remote_endpoint=""
    remote_path=""
    embeddings_dim=0
    instance = None

    def __init__(self, local_path="", remote_endpoint="", remote_path="", embeddings_dim=300):
        """
        Classe permettant de gérer fasttext et de ne le charger qu'une fois

        L'init permet de parametrer le stockage (local et remote)
        pour vérifier que les poids sont disponibles localement, ou les récupérer, utiliser get_model_files
        pour charger, utiliser load_model.
        pour le décharger de la mémoire, utiliser delete_model

        IMPORTANT: le redimensionnement plante sur certaines machines...

        :param local_path: path où sauvegarder/charger localement les poids
        :param remote_endpoint: s3fs endpoint pour minio
        :param remote_path: path où sauvegarder/charger les poids sur minio
        :param embeddings_dim: dimension finale des embeddings (par défaut: 300)
        """
        if not FasttextSingleton.local_path:
            FasttextSingleton.local_path = local_path
        if not FasttextSingleton.remote_endpoint:
            FasttextSingleton.remote_endpoint = remote_endpoint
        if not FasttextSingleton.remote_path:
            FasttextSingleton.remote_path = remote_path
        if not FasttextSingleton.embeddings_dim:
            FasttextSingleton.embeddings_dim = embeddings_dim

    def load_model(self):
        """
        Charger le modèle en mémoire si pas encore chargé
        """
        if not FasttextSingleton.instance:
            FasttextSingleton.instance = FasttextSingleton.__fasttext(FasttextSingleton.local_path,
                                                                      FasttextSingleton.embeddings_dim)

    def delete_model(self):
        """
        Supprimer le modèle de la mémoire
        """
        if FasttextSingleton.instance is not None:
            del FasttextSingleton.instance
            FasttextSingleton.instance = None

    def __getattr__(self, name):
        """
        passage des autres méthodes au modèle fasttext
        """
        return getattr(self.instance.ft, name)

    def get_embeddings(self, word):
        """
        Récupérer les embeddings pour un mot

        :param word: mot dont on veut les embeddings
        :returns: liste de dimension (embeddings_size,)
        """
        return self.instance.ft[word]

    def get_model_files(self):
        """
        Vérifier que les fichiers sont localement disponibles.
        Sinon, on tente de les récupérer sur minio.
        En dernier recours, on les retélécharge via la lib fasttext
        """
        logger = logging.getLogger('FasttextSingleton')
        if not os.path.exists(FasttextSingleton.local_path) \
                or len(os.listdir(FasttextSingleton.local_path)) == 0:
            logger.info('fasttext in config and model not found')
            if FasttextSingleton.remote_endpoint:
                fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': FasttextSingleton.remote_endpoint})
                logger.info('trying to load from minio')
                try:
                    fs.get(FasttextSingleton.remote_path, 
                           FasttextSingleton.local_path,
                           recursive=True)
                    if len(os.listdir(FasttextSingleton.local_path)) == 0:
                        logger.error('error downloading fasttext from minio')
                except:
                    pass
        if not os.path.exists(FasttextSingleton.local_path) \
                or len(os.listdir(FasttextSingleton.local_path)) == 0:
            logger.info('downloading from lib')
            os.makedirs(FasttextSingleton.local_path, exist_ok=True)
            fasttext.util.download_model('fr', 'strict')
            shutil.move('cc.fr.300.bin', 
                        os.path.join(FasttextSingleton.local_path, 'cc.fr.300.bin'))
            if FasttextSingleton.remote_endpoint:
                fs = s3fs.S3FileSystem(client_kwargs={'endpoint_url': FasttextSingleton.remote_endpoint})
                try:
                    fs.put(FasttextSingleton.local_path, 
                        FasttextSingleton.remote_path, recursive=True)
                except Exception as e:
                    logger.error(f'Error pushing fasttext on minio : {e}')
