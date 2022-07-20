import os
import pandas as pd
import yaml
import logging

import nltk 
nltk.download('stopwords')

from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer


from .embedding_dictionary import EmbeddingDictionary
from . import preprocessing as pre


class Cleaner:
    
    def __init__(self, bdd, save_path_no_ext, 
                 use_stemmer=True,
                 use_fasttext=False,
                 ngrams_size=None):
        """
        Classe Mère permettant de 
            - récupérer les données depuis la base, 
            - calculer les champs consolidés
            - Filtrer les champs vides
            - calculer le dictionnaire de données
            - encoder toutes les données avec le dictionnaire
            
        Elle ne peut être utilisée telle qu'elle, une sous-classe doit être utilisée

        :param bdd: base PostGre_SQL_DB à utiliser
        :param save_path_no_ext: chemin pour sauvegarder/charger sans ext
        :param use_stemmer: utilisation ou non d'un stemmer
        :param use_fasttext: bool
        :param ngrams_size: si True, on encode en ngrams. sinon, en mots
        """
        self.bdd = bdd
        self.save_path_no_ext = save_path_no_ext
        self.sql_file = self.get_sqlfile_path(save_path_no_ext)
        self.raw_data_file = self.get_rawdata_path(save_path_no_ext)
        self.data_dict_file = save_path_no_ext + "_data_dict"
        self.prepared_data_file = save_path_no_ext + "_data_prepared.csv"
        
        self.use_stemmer = use_stemmer
        self.stopwords = nltk.corpus.stopwords.words('french')
        self.stemmer = None
        if self.use_stemmer:
            self.stemmer = SnowballStemmer(language='french')
        self.use_fasttext = use_fasttext
        self.ngrams_size = ngrams_size
        self.logger = logging.getLogger(self.__class__.__name__)
        
    @classmethod
    def get_sqlfile_path(cls, save_path_no_ext):
        return save_path_no_ext + "_sql_request.yaml"
    
    @classmethod
    def get_rawdata_path(cls, save_path_no_ext):
        return save_path_no_ext + "_data.csv"
        
    def get_and_clean_data(self, sql_request, nomenclature):
        """
        Récupération des données (ou chargement si disponibles)
        Calcul des champs consolidés actet_c_libelle et prof_x
        Filtrage des actet_x et actet_c_libelle vides
        enregistrement dans self.raw_data_file

        :param sql_request: requète SQL pour récupérer les données
        :param nomenclature: Nomenclature à utiliser
        :returns: str, chemin du fichier des données consolidées
        """
        if os.path.exists(self.sql_file) and os.path.exists(self.raw_data_file):
            with open(self.sql_file) as f:
                old_req = yaml.load(f, Loader=yaml.FullLoader)
            if old_req['nomenclature'] == nomenclature.name \
                    and old_req['sql'] == sql_request \
                        and old_req['stemmer'] == self.use_stemmer \
                            and old_req['ngrams_size'] == self.ngrams_size \
                                and old_req['fasttext'] == self.use_fasttext:
                self.logger.info('data already extracted, skipping.')
                return
            else:
                raise ValueError('SQL request does not correspond to the file found on disk')
        
        # clean df
        for df in self.bdd.read_from_sql_with_chunksize(sql_request, chunksize=10000):
            df_no_empty = self.clean_bi_df(nomenclature, df)

            with open(self.raw_data_file, 'a') as f:
                df_no_empty.to_csv(f, mode='a', sep=';', header=f.tell()==0, index=False)

        # clean nomenclature
        for k in nomenclature.nodes.keys():
            nomenclature.clean_node(k, self.clean_bi_field)

        with open(self.sql_file, 'w') as f:
            yaml.dump({
                'class': self.__class__.__name__,
                'nomenclature': nomenclature.name,
                'sql': sql_request,
                'stemmer': self.use_stemmer,
                'ngrams_size': self.ngrams_size,
                'fasttext': self.use_fasttext
            }, f)

        return self.raw_data_file

    def clean_bi_df(self, nomenclature, df):
        """
        Calcul des champs consolidés actet_c_libelle et prof_x
        Filtrage des actet_x et actet_c_libelle vides

        :param nomenclature: Nomenclature utilisée
        :param df: pd.Dataframe à traiter
        :returns: pd.Dataframe
        """
        raise NotImplementedError()

    def clean_bi_field(self, str_value):
        """
        Nettoyage d'un seul champs: cleaning et application éventuelle d'un stemmer
        
        :param str_value: string à nettoyer
        :return: str
        """
        if not str_value:
            return None
        ret_value = pre.clean(str_value)
        ret_value = ' '.join([word for word in ret_value.split() if word not in (self.stopwords)])
        if self.use_stemmer:
            ret_value = " ".join([self.stemmer.stem(word) for word in ret_value.split(" ")])
        if not ret_value or ret_value == " ":
            ret_value = None

        return ret_value

    def create_dictionary(self, nomenclature):
        """
        Création du dictionnaire des données

        :param nomenclature: Nomenclature utilisée
        :returns: str, chemin du dictionnaire
        """
        embedding_voc_class = EmbeddingDictionary.factory(use_ngrams=(self.ngrams_size is not None))
        embeddings_voc = embedding_voc_class(pad_token_at_start=True, 
                                             distinct_null_token=True,
                                             use_fasttext=self.use_fasttext,
                                             ngrams_size=self.ngrams_size)
        embeddings_voc.add_data([n.desc_cleaned for n in nomenclature.nodes.values()])
        
        for df in pd.read_csv(self.raw_data_file, chunksize=60000, dtype=str, sep=";"):  
            embeddings_voc.add_data(self.get_df_corpus_tokens_for_vocabulary(df))
                
        embeddings_voc.finalize_dict()
        embeddings_voc.save(self.data_dict_file)

        return self.data_dict_file
    
    def get_df_corpus_tokens_for_vocabulary(self, df_input):
        """
        Extrait l'ensemble des tokens dans df_input à ajouter au corpus 
        existant 
        
        :param df_input: dataframe
        :returns: liste de tokens
        """
        raise NotImplementedError()
        
    def put_training_data_in_voc(self):
        """
        Encodage de toutes les données avec le dictionnaire
        Tout est chargé depuis le disque et sauvé sur le disque

        :returns: str, chemin du fichier de données encodées
        """
        # TODO : read existing if data not regenerated
        embeddings_voc = EmbeddingDictionary.load(self.data_dict_file)
        
        for df in pd.read_csv(self.raw_data_file, chunksize=60000, dtype=str, sep=";"):
            df = self.put_data_in_voc(embeddings_voc, df)
            with open(self.prepared_data_file, 'a') as f:
                df.to_csv(f, mode='a', sep=';', header=f.tell()==0,index=False)
        
        return self.prepared_data_file

    def put_data_in_voc(self, embeddings_voc, df):
        """
        Encodage d'une dataframe

        :param embeddings_voc: EmbeddingsVoc à utiliser
        :param df:pd.Dataframe à encoder
        :returns: pd.Dataframe encodée
        """
        raise NotImplementedError()
        
    @classmethod
    def create_factory(cls, string):
        """
        Factory permettant la création d'une classe-fille à partir de son
        nom
        
        :param string: nom de la classe
        :returns: classe fille (à instancier)
        """
        return next(c for c in cls.__subclasses__() if c.__name__.lower() == string.lower())
    
    @classmethod
    def load_factory(cls, save_path_no_ext):
        """
        Permet de charger une classe fille depuis le disque en gérant l'instantiation
        de la bonne classe
        
        :param save_path_no_ext: dossier de sauvegarde
        :returns: Cleaner instancié
        """
        sql_file = save_path_no_ext + "_sql_request.yaml"
        if os.path.exists(sql_file):
            with open(sql_file) as f:
                save_req = yaml.load(f, Loader=yaml.FullLoader)
                if 'class' not in save_req:
                    # retrocompatibilité
                    return NafCleaner
                return next(c for c in cls.__subclasses__() 
                                if c.__name__.lower() == save_req['class'].lower())
        
        raise ValueError(f'{sql_file} not found')


class NafCleaner(Cleaner):
    
    def __init__(self, bdd, save_path_no_ext, 
                 use_stemmer=True,
                 use_fasttext=False,
                 ngrams_size=None):
        """
        Classe permettant de 
            - récupérer les données depuis la base, 
            - calculer les champs consolidés actet_c_libelle et prof_x
            - Filtrage des actet_x et actet_c_libelle vides
            - calculer le dictionnaire de données
            - encoder toutes les données avec le dictionnaire dans actet_repr, rs_repr, prof_repr

        :param bdd: base PostGre_SQL_DB à utiliser
        :param save_path_no_ext: chemin pour sauvegarder/charger sans ext
        :param use_stemmer: utilisation ou non d'un stemmer
        :param use_fasttext: bool
        :param ngrams_size: si True, on encode en ngrams. sinon, en mots
        """
        super().__init__(bdd, save_path_no_ext, 
                         use_stemmer=use_stemmer,
                         use_fasttext=use_fasttext,
                         ngrams_size=ngrams_size)
        self.logger = logging.getLogger(self.__class__.__name__)

    def clean_bi_df(self, nomenclature, df):
        """
        Calcul des champs consolidés actet_c_libelle et prof_x
        Filtrage des actet_x et actet_c_libelle vides

        :param nomenclature: Nomenclature utilisée
        :param df: pd.Dataframe à traiter
        :returns: pd.Dataframe
        """
        prof_libelle = []
        for _, line in df[['profs_x', 'profi_x', 'profa_x']].iterrows():
            if line.profs_x:
                prof_libelle.append(line.profs_x)
            elif line.profi_x:
                prof_libelle.append(line.profi_x)
            elif line.profa_x:
                prof_libelle.append(line.profa_x)
            else:
                prof_libelle.append('')
        
        df['prof_x'] = prof_libelle
        df = df.drop(['profs_x', 'profi_x', 'profa_x'], axis=1)

        # clean, remove stop words, stem
        for col in ['actet_x', 'rs_x', 'prof_x']:
            df[col] = df[col].map(lambda x: self.clean_bi_field(x))

        # Filter by non null value in actet
        df_no_empty = df[df['actet_x'] != '' ]
        df_no_empty = df_no_empty[df_no_empty['actet_x'].notna()]
        if 'actet_c' in df_no_empty.columns:
            nom_code_exists = df_no_empty['actet_c'].map(lambda x : x in nomenclature.nodes)
            df_no_empty = df_no_empty[nom_code_exists]
        df_no_empty = df_no_empty.reset_index()
        return df_no_empty
    
    def get_df_corpus_tokens_for_vocabulary(self, df_input):
        """
        Encodage de toutes les données avec le dictionnaire
        Tout est chargé depuis le disque et sauvé sur le disque

        :returns: str, chemin du fichier de données encodées
        """
        return df_input['actet_x'].to_numpy(dtype="str").tolist() \
               + df_input['rs_x'].to_numpy(dtype="str").tolist() \
               + df_input['prof_x'].to_numpy(dtype="str").tolist()

    def put_data_in_voc(self, embeddings_voc, df):
        """
        Encodage d'une dataframe

        :param embeddings_voc: EmbeddingsVoc à utiliser
        :param df:pd.Dataframe à encoder
        :returns: pd.Dataframe encodée
        """
        df['actet_repr'] = [embeddings_voc.convert_voc(str(v)) for v in df['actet_x']]
        df['rs_repr'] = [embeddings_voc.convert_voc(str(v)) for v in df['rs_x']]
        df['prof_repr'] = [embeddings_voc.convert_voc(str(v)) for v in df['prof_x']]
        df = df.drop(['actet_x', 'rs_x', 'prof_x'], axis=1)
        return df


class ProfCleaner(Cleaner):
    
    def __init__(self, bdd, save_path_no_ext, 
                 use_stemmer=True,
                 use_fasttext=False,
                 ngrams_size=None):
        """
        Classe permettant de 
            - récupérer les données depuis la base, 
            - calculer les champs consolidés prof_x, prof_c, prof_c_libelle
            - Filtrage des prof_x et prof_c vides
            - calculer le dictionnaire de données
            - encoder toutes les données avec le dictionnaire dans actet_repr, rs_repr, prof_repr

        :param bdd: base PostGre_SQL_DB à utiliser
        :param save_path_no_ext: chemin pour sauvegarder/charger sans ext
        :param use_stemmer: utilisation ou non d'un stemmer
        :param use_fasttext: bool
        :param ngrams_size: si True, on encode en ngrams. sinon, en mots
        """
        super().__init__(bdd, save_path_no_ext, 
                         use_stemmer=use_stemmer,
                         use_fasttext=use_fasttext,
                         ngrams_size=ngrams_size)
        self.logger = logging.getLogger(self.__class__.__name__)

    def clean_bi_df(self, nomenclature, df):
        """
        Calcul des champs consolidés actet_c_libelle et prof_x
        Filtrage des actet_x et actet_c_libelle vides

        :param nomenclature: Nomenclature utilisée
        :param df: pd.Dataframe à traiter
        :returns: pd.Dataframe
        """
        prof_libelle = []
        prof_classe = []
        for i, line in df[['profs_x', 'profi_x', 'profa_x']].iterrows():
            line_gt = None
            if 'profs_c' in df.columns:
                line_gt = df[['profs_c', 'profi_c', 'profa_c']].iloc[i]
            if line.profs_x:
                prof_libelle.append(line.profs_x)
                if line_gt is not None:
                    prof_classe.append(line_gt.profs_c)
            elif line.profi_x:
                prof_libelle.append(line.profi_x)
                if line_gt is not None:
                    prof_classe.append(line_gt.profi_c)
            elif line.profa_x:
                prof_libelle.append(line.profa_x)
                if line_gt is not None:
                    prof_classe.append(line_gt.profa_c)
            else:
                prof_libelle.append('')
                if line_gt is not None:
                    prof_classe.append('')
        
        df['prof_x'] = prof_libelle
        if len(prof_classe):
            df['prof_c'] = prof_classe

        df = df.drop(['profs_x', 'profi_x', 'profa_x'], axis=1)
        if 'profs_c' in df.columns:
            df = df.drop(['profs_c', 'profi_c', 'profa_c'], axis=1)

        # clean, remove stop words, stem
        for col in ['actet_x', 'rs_x', 'prof_x']:
            df[col] = df[col].map(lambda x: self.clean_bi_field(x))

        # Filter by non null value in actet
        df_no_empty = df[df['prof_x'] != '' ]
        df_no_empty = df_no_empty[df_no_empty['prof_x'].notna()]
        if 'prof_c' in df_no_empty.columns:
            nom_code_exists = df_no_empty['prof_c'].map(lambda x : x in nomenclature.nodes)
            df_no_empty = df_no_empty[nom_code_exists]
        df_no_empty = df_no_empty.reset_index()
        return df_no_empty
    
    def get_df_corpus_tokens_for_vocabulary(self, df_input):
        """
        Extrait l'ensemble des tokens dans df_input à ajouter au corpus 
        existant 
        
        :param df_input: dataframe
        :returns: liste de tokens
        """
        return df_input['actet_x'].to_numpy(dtype="str").tolist() \
               + df_input['rs_x'].to_numpy(dtype="str").tolist() \
               + df_input['prof_x'].to_numpy(dtype="str").tolist()

    def put_data_in_voc(self, embeddings_voc, df):
        """
        Encodage d'une dataframe

        :param embeddings_voc: EmbeddingsVoc à utiliser
        :param df:pd.Dataframe à encoder
        :returns: pd.Dataframe encodée
        """
        df['actet_repr'] = [embeddings_voc.convert_voc(str(v)) for v in df['actet_x']]
        df['rs_repr'] = [embeddings_voc.convert_voc(str(v)) for v in df['rs_x']]
        df['prof_repr'] = [embeddings_voc.convert_voc(str(v)) for v in df['prof_x']]
        df = df.drop(['actet_x', 'rs_x', 'prof_x'], axis=1)
        return df
