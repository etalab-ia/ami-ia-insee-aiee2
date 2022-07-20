import os, sys
import abc
import pickle
import logging
import numpy as np
from nltk import ngrams

from .preprocessing import preprocessing_no_stemmer
from .fasttext_singleton import FasttextSingleton

# Create vocabulary dictionary

class EmbeddingDictionary:
    
    def __init__(self, pad_token_at_start=True, distinct_null_token=False, use_fasttext=False):
        """
        Classe mère abstraite permettant l'extraction d'un dictionnaire d'encodage des données.
        Si fasttext est utilisé, il extrait aussi les vecteurs embeddings correspondant.

        :param pad_token_at_start: si True, pad_token = 0, sinon c'est le plus haut indice du voc
        :param distinct_null_token: si True, on le crée à la fin, sinon il vaut pad_token
        :param use_fasttext: si True, on utilise fasttext
        """
        self.dico_vocab = {}
        self.pad_token_at_start = pad_token_at_start
        self.distinct_null_token = distinct_null_token
        self.nb_docs_analyzed = 0
        self.nb_tokens_analyzed = 0
        self.nb_different_tokens = 0
        self.pad_token = 0
        self.null_token = -1
        self.finalized = False
        
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def get_vocab_size(self):
        """
        Renvoie la taille du vocabulaire
        
        :return: int
        """
        return self.nb_different_tokens
    
    def add_data(self, corpus_part):
        """
        Ajoute de la data à analyser

        :param corpus_part: ensemble de documents. Doit être itérable
        """
        start_token = 0
        if self.pad_token_at_start:
            start_token = 1

        self.logger.info(f'Adding docs {self.nb_docs_analyzed} to {self.nb_docs_analyzed + len(corpus_part)}')
        for doc in corpus_part:
            self.nb_docs_analyzed += 1
            if doc is None:
                continue
            tokens = self.split_doc_in_tokens(doc)
            self.nb_tokens_analyzed += len(tokens)
            for t in tokens:
                if t not in self.dico_vocab:
                    self.dico_vocab[t] = len(self.dico_vocab) + start_token

    @abc.abstractmethod
    def split_doc_in_tokens(self, doc):
        """
        Fonction qui transforme un document (une string) en liste de tokens
        Doit être implémentée dans les classes filles
        
        :param doc: string
        :return: list of token
        """
        raise NotImplementedError
                 
    def finalize_dict(self):
        """
        Gère les token de padding et null
        Gère l'extraction fasttext si besoin
        
        :return: None
        """
        self.nb_different_tokens = len(self.dico_vocab)

        if self.pad_token_at_start:
            self.pad_token = 0
        else:
            self.pad_token = len(self.dico_vocab) + 1
        self.nb_different_tokens += 1
        
        if not self.distinct_null_token:
            self.null_token = self.pad_token
        else:
            if self.pad_token == 0:
                self.null_token = len(self.dico_vocab) + 1
                self.nb_different_tokens += 1
            else:
                self.null_token = 0
                
        self.logger.info(f'Finalized: {self}')
        self.finalized = True
        
    def __str__(self):
        _str = f'nb_docs = {self.nb_docs_analyzed}, nb_tokens: {self.nb_tokens_analyzed}, '
        _str += f'nb_unique_tokens: {self.nb_different_tokens}'
        return _str
        
    def build_dict(self, corpus):
        """
        Construit et finalise le vocabulaire, si corpus est entier

        :param corpus: ensemble de documents. Doit être itérable
        """
        self.add_data(corpus)
        self.finalize_dict()
        
    def save(self, file_path_no_ext):
        """
        Sauvegarde dans file_path_no_ext.pkl
        si use_fasttext, sauvegarde les embeddings extraits dans file_path_no_ext_ft_vectors.npy

        :param file_path_no_ext: chemin de sauvegarde, sans ext
        """
        if not self.finalized:
            raise RuntimeError('EmbeddingDictionary not finalized')
        repr_dict = {
            'type': self.__class__.__name__,
            'dico_vocab': self.dico_vocab,
            'pad_token_at_start': self.pad_token_at_start,
            'distinct_null_token': self.distinct_null_token,
            'nb_docs_analyzed': self.nb_docs_analyzed,
            'nb_tokens_analyzed': self.nb_tokens_analyzed,
            'nb_different_tokens': self.nb_different_tokens,
            'pad_token': self.pad_token,
            'null_token': self.null_token,
        }
        with open(file_path_no_ext + '.pkl', 'wb') as f:
            pickle.dump(repr_dict, f)
            
    @staticmethod
    def load(file_path_no_ext):
        """
        charge depuis file_path_no_ext.pkl puis instancie la bonne classe fille

        :param file_path_no_ext: chemin de chargement, sans ext
        :returns: EmbeddingDictionary
        """
        with open(file_path_no_ext + '.pkl', 'rb') as f:
            repr_dict = pickle.load(f)

        if 'type' not in repr_dict:
            # backward compatibility
            return WordDictionary.load(file_path_no_ext)
        return eval(repr_dict['type']).load(file_path_no_ext)

    def convert_voc(self, doc):
        """
        convertit un doc en liste d'entier, indices des mots dans le dictionnaire
        
        IMPORTANT : les OOV sont enlevés

        :param doc: document à convertir
        :returns: liste d'entiers
        """
        if not self.finalized:
            raise RuntimeError('EmbeddingDictionary not finalized')
        if doc is None or not len(doc):
            return [self.null_token]
        tokens = self.split_doc_in_tokens(doc)
        # if token not in vocab, we remove it
        return [self.dico_vocab[t] for t in tokens if t in self.dico_vocab]

    @staticmethod
    def factory(use_ngrams=False):
        """
        factory méthode pour l'instanciation de classes filles
        
        :param use_ngrams: si True, renvoie NGramsDictionary. Sinon WordDictionary
        :return: classe fille non instanciée
        """
        if use_ngrams:
            return NGramsDictionary
        return WordDictionary

class WordDictionary(EmbeddingDictionary):

    def __init__(self, pad_token_at_start=True, distinct_null_token=False, use_fasttext=False, **kwargs):
        """
        classe de dictionnaire pour les représentations à base de mot
        
        :param pad_token_at_start: si True, pad_token = 0, sinon c'est le plus haut indice du voc
        :param distinct_null_token: si True, on le crée à la fin, sinon il vaut pad_token
        :param use_fasttext: si True, on utilise fasttext
        :param kwargs: paramêtres non utilisés
        """
        super().__init__(pad_token_at_start, distinct_null_token)

        self.use_fasttext = use_fasttext
        # compteur de mots du corpus non connu de fasttext 
        # (mais fasttext produit quand meme des embeddings dans ce cas)
        self.nb_ft_oov = 0
        # embeddings fasttext pour les mots du corpus
        self.ft_vectors = None

    def split_doc_in_tokens(self, doc):
        """
        découpe la str en tokens sur les espaces
        """
        return preprocessing_no_stemmer(doc).split(" ")

    def personnalize_fasttext(self):
        """
        charge fasttext, et extrait les embeddings correspondant au vocabulaire contenu
        dans les documents analysés.
        compte les oov.
        """
        self.logger.info(f'Extracting Fasttext embeddings')
        ft = FasttextSingleton()
        ft.load_model()
        self.nb_ft_oov = 0
        self.ft_vectors = [np.zeros((ft.get_dimension()))]
        for t in self.dico_vocab.keys():
            if t not in ft.words:
                self.nb_ft_oov += 1
            self.ft_vectors.append(ft.get_embeddings(t))

        self.ft_vectors = np.stack(self.ft_vectors)
           
    def finalize_dict(self):
        """
        Gère les token de padding et null
        Gère l'extraction fasttext si besoin
        """
        self.nb_different_tokens = len(self.dico_vocab)

        if self.use_fasttext:
            self.personnalize_fasttext()
        
        if self.pad_token_at_start:
            self.pad_token = 0
        else:
            self.pad_token = len(self.dico_vocab) + 1
            if self.use_fasttext:
                self.ft_vectors = [self.ft_vectors, np.zeros((1, self.ft_vectors.shape[1]))]
                self.ft_vectors = np.concatenate(self.ft_vectors)
        self.nb_different_tokens += 1
        
        if not self.distinct_null_token:
            self.null_token = self.pad_token
        else:
            if self.pad_token == 0:
                self.null_token = len(self.dico_vocab) + 1
                if self.use_fasttext:
                    self.ft_vectors = [self.ft_vectors, np.zeros((1, self.ft_vectors.shape[1]))]
                    self.ft_vectors = np.concatenate(self.ft_vectors)
                self.nb_different_tokens += 1
            else:
                self.null_token = 0
                
        self.logger.info(f'Finalized: {self}')
        self.finalized = True

    def __str__(self):
        _str = EmbeddingDictionary.__str__(self)
        if self.use_fasttext:
            _str += f', nb_fasttext_oov: {self.nb_ft_oov}'
        return _str

    def save(self, file_path_no_ext):
        """
        Sauvegarde dans file_path_no_ext.pkl
        si use_fasttext, sauvegarde les embeddings extraits dans file_path_no_ext_ft_vectors.npy

        :param file_path_no_ext: chemin de sauvegarde, sans ext
        """
        if not self.finalized:
            raise RuntimeError('EmbeddingDictionary not finalized')
        repr_dict = {
            'type': self.__class__.__name__,
            'dico_vocab': self.dico_vocab,
            'pad_token_at_start': self.pad_token_at_start,
            'distinct_null_token': self.distinct_null_token,
            'nb_docs_analyzed': self.nb_docs_analyzed,
            'nb_tokens_analyzed': self.nb_tokens_analyzed,
            'nb_different_tokens': self.nb_different_tokens,
            'pad_token': self.pad_token,
            'null_token': self.null_token,
            'use_fasttext': self.use_fasttext,
            'nb_ft_oov': self.nb_ft_oov,
            'ft_vectors': ""
        }
        if self.ft_vectors is not None:
            np.save( file_path_no_ext + '_ft_vectors.npy', self.ft_vectors)
            repr_dict['ft_vectors'] = os.path.basename(file_path_no_ext) + '_ft_vectors.npy'
        with open(file_path_no_ext + '.pkl', 'wb') as f:
            pickle.dump(repr_dict, f)

    @staticmethod
    def load(file_path_no_ext):
        """
        charge depuis file_path_no_ext.pkl
        si use_fasttext, charge les embeddings extraits depuis file_path_no_ext_ft_vectors.npy

        :param file_path_no_ext: chemin de chargement, sans ext
        :returns: EmbeddingDictionary
        """
        with open(file_path_no_ext + '.pkl', 'rb') as f:
            repr_dict = pickle.load(f)
        
        emb_dic = WordDictionary(pad_token_at_start=repr_dict['pad_token_at_start'], 
                                 distinct_null_token=repr_dict['distinct_null_token'],
                                 use_fasttext=repr_dict['use_fasttext'])
        emb_dic.dico_vocab = repr_dict['dico_vocab']
        emb_dic.nb_docs_analyzed = repr_dict['nb_docs_analyzed']
        emb_dic.nb_tokens_analyzed = repr_dict['nb_tokens_analyzed']
        emb_dic.nb_different_tokens = repr_dict['nb_different_tokens']
        emb_dic.pad_token = repr_dict['pad_token']
        emb_dic.null_token = repr_dict['null_token']
        emb_dic.nb_ft_oov = repr_dict['nb_ft_oov']
        if repr_dict['ft_vectors']:
            emb_dic.ft_vectors = np.load(file_path_no_ext + '_ft_vectors.npy')
        emb_dic.finalized = True
        emb_dic.logger.info(f'Loaded: {emb_dic}')
        return emb_dic

    def get_embeddings(self):
        """
        si use_fasttext, renvoie la matrice des embeddings (nb_words, embeddings_size)
        """
        if self.ft_vectors is None:
            raise RuntimeError('Not in fasttext mode')
        return self.ft_vectors


class NGramsDictionary(EmbeddingDictionary):

    def __init__(self, pad_token_at_start=True, distinct_null_token=False, ngrams_size=2, **kwargs):
        """
        Classe de dictionnaire pour les représentations ngrams
        
        :param pad_token_at_start: si True, pad_token = 0, sinon c'est le plus haut indice du voc
        :param distinct_null_token: si True, on le crée à la fin, sinon il vaut pad_token
        :param ngrams_size: taille de ngram à appliquer
        :param kwargs: paramêtres non utilisés
        """
        super().__init__(pad_token_at_start, distinct_null_token)

        self.ngrams_size = ngrams_size

    def split_doc_in_tokens(self, doc):
        """
        découpe la str en tokens de N lettres.
        On pad le début et la fin de chaque mot par #
        """
        words = preprocessing_no_stemmer(doc)
        ngram_list = list(ngrams(words, self.ngrams_size, 
                                 pad_left=True, left_pad_symbol=' ',
                                 pad_right=True, right_pad_symbol=' '))
        ngram_list = [''.join(ngram).replace(' ', '#') for ngram in ngram_list]
        return ngram_list

    def save(self, file_path_no_ext):
        """
        Sauvegarde dans file_path_no_ext.pkl
        si use_fasttext, sauvegarde les embeddings extraits dans file_path_no_ext_ft_vectors.npy

        :param file_path_no_ext: chemin de sauvegarde, sans ext
        """
        if not self.finalized:
            raise RuntimeError('EmbeddingDictionary not finalized')
        repr_dict = {
            'type': self.__class__.__name__,
            'dico_vocab': self.dico_vocab,
            'pad_token_at_start': self.pad_token_at_start,
            'distinct_null_token': self.distinct_null_token,
            'nb_docs_analyzed': self.nb_docs_analyzed,
            'nb_tokens_analyzed': self.nb_tokens_analyzed,
            'nb_different_tokens': self.nb_different_tokens,
            'pad_token': self.pad_token,
            'null_token': self.null_token,
            'ngrams_size': self.ngrams_size
        }
        with open(file_path_no_ext + '.pkl', 'wb') as f:
            pickle.dump(repr_dict, f)

    @staticmethod
    def load(file_path_no_ext):
        """
        charge depuis file_path_no_ext.pkl
        si use_fasttext, charge les embeddings extraits depuis file_path_no_ext_ft_vectors.npy

        :param file_path_no_ext: chemin de chargement, sans ext
        :returns: EmbeddingDictionary
        """
        with open(file_path_no_ext + '.pkl', 'rb') as f:
            repr_dict = pickle.load(f)
        
        emb_dic = NGramsDictionary(pad_token_at_start=repr_dict['pad_token_at_start'], 
                                   distinct_null_token=repr_dict['distinct_null_token'],
                                   ngrams_size=repr_dict['ngrams_size'])
        emb_dic.dico_vocab = repr_dict['dico_vocab']
        emb_dic.nb_docs_analyzed = repr_dict['nb_docs_analyzed']
        emb_dic.nb_tokens_analyzed = repr_dict['nb_tokens_analyzed']
        emb_dic.nb_different_tokens = repr_dict['nb_different_tokens']
        emb_dic.pad_token = repr_dict['pad_token']
        emb_dic.null_token = repr_dict['null_token']
        emb_dic.finalized = True
        emb_dic.logger.info(f'Loaded: {emb_dic}')
        return emb_dic
