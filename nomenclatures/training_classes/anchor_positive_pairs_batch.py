from tensorflow import keras
import numpy as np
import json

import os, sys
sys.path.append(os.path.join('..', '..'))
from data_import.bdd import PostGre_SQL_DB
from .nomenclature_distance import NomenclatureDistance

class AnchorPositivePairsBatch(keras.utils.Sequence):
    def __init__(self, nomenclature_distance: NomenclatureDistance, 
                 seq_len):
        """
        Classe permettant de formatter les inputs et de les batcher par pairs correspondantes

        :param nomenclature_distance: la NomenclatureDistance construite avec les data
        :param seq_len: la longueur maximale de la séquence totale de texte
        """
        # input data
        self.input_classes = None
        self.input_fields = None
        # input functions
        self.nomenclature_distance = nomenclature_distance
        # batching variables
        self.seq_len = seq_len
        self.batch_size = 0
        # epoch variables
        self.num_batchs = 0
        self.current_pos = 0

    def set_data(self,  input_classes, input_fields, num_batchs, batch_size):
        """
        setter des données à utiliser

        :param input_classes: données "label" (codes de la nomenclature)
        :param input_fields: données "input" (des BIs)
        :param num_batchs: nombre de batch à servir
        :param batch_size: taille des batch
        """
        self.input_classes = input_classes
        self.input_fields = input_fields
        self.num_batchs = num_batchs
        self.batch_size = batch_size
        
    def __len__(self):
        return self.num_batchs
    
    def get_nomenclature(self):
        """
        :returns: la Nomenclature de self.nomenclature_distance
        """
        return self.nomenclature_distance.nomenclature
    
    def get_embeddings_voc(self):
        """
        :returns: l'EmbeddingsDictionary de self.nomenclature_distance.nomenclature
        """
        return self.nomenclature_distance.nomenclature.embeddings_dict
    
    @staticmethod
    def pad_seq(seq, seq_len: int, pad_token=0):
        """
        padde une séquence

        :param seq: liste à padder
        :param seq_len: longueur à atteindre
        :param pad_token: int à utiliser pour padder
        :returns : la séquence paddée
        """
        return seq + [pad_token for _ in range(seq_len - len(seq))]
    
    @staticmethod
    def get_fields(*args, start_ind=1):
        """
        construit une liste contenant len(args[0]) fois start_ind, 
        puis len(args[1]) fois start_ind+1, etc

        :param args: liste de liste
        :param start_ind: 1e valeur de la liste
        :returns: liste d'int
        """
        fields = []
        for i, f in enumerate(args):
            fields += [start_ind+i for _ in f]
        return fields
    
    @staticmethod
    def get_positions(*args, start_ind=1):
        """
        construit une liste contenant range(start_ind, len(args[0])+start_ind), 
        puis range(start_ind, len(args[1])+start_ind), etc

        :param args: liste de liste
        :param start_ind: 1e valeur de la liste
        :returns: liste d'int
        """
        positions=[]
        for f in args:
            positions += list(range(start_ind, len(f)+start_ind))
        return positions
    
    def format_input(self, *args):
        """
        construit 3 listes à partir de :param args: :
            1 - concaténation des listes de args, paddée à seq_length
            2 - liste créée par :method get_fields:,  paddée à seq_length
            3 - liste créée par :method get_positions:,  paddée à seq_length

        Ces listes sont l'input attendu par le réseau en mode prédiction

        :param args: liste de liste
        :returns: les 3 listes, dans l'ordre
        """
        input_seq = self.pad_seq(sum(args, []), self.seq_len, 
                                 self.nomenclature_distance.nomenclature.embeddings_dict.pad_token)
        fields = self.get_fields(*args)
        fields_seq = self.pad_seq(fields, self.seq_len, 
                                  self.nomenclature_distance.nomenclature.embeddings_dict.pad_token)
        positions_seq = self.pad_seq(self.get_positions(*args), self.seq_len, 
                                     self.nomenclature_distance.nomenclature.embeddings_dict.pad_token)
        return input_seq, fields_seq, positions_seq

    def __getitem__(self, _idx):
        """
        itérateur pour le training

        construit un batch de taille (7, self.batch_size, self.seq_len) dans lequel 
            - (0, :), (1, :), (2, :) = self.format_input(input_fields)
            - (3, :), (4, :), (5, :) = self.format_input des données de nomenclature
            - (6, :) = liste des codes nomenclature des exemples du batch

        :returns: un batch
        """
        x = np.empty((7, self.batch_size, self.seq_len), dtype=np.float32)
        for i in range(self.batch_size):
            class_id = self.input_classes[self.current_pos].tolist()[0]
            # format data values
            data_values = self.input_fields[self.current_pos].tolist()
            x[0, i], x[1, i], x[2, i] = self.format_input(*data_values)
            # format label values
            positive_data = self.nomenclature_distance.nomenclature.get_nomenclature_embeddings(class_id)
            x[3, i], x[4, i], x[5, i] = self.format_input(positive_data)
            # format class id
            x[6, i] = self.pad_seq([self.nomenclature_distance.nodes_index[class_id]], self.seq_len, 
                                   self.nomenclature_distance.nomenclature.embeddings_dict.pad_token)
            
            self.current_pos += 1
            self.current_pos %= len(self.input_fields)
        return x

    def save(self, file_name_no_ext):
        """
        Sauve la nomenclature_distance vers file_name_no_ext + '_nomdist',
        et self vers file_name_no_ext.json

        :param file_name_no_ext: path de sauvegarde sans extension
        """
        self.nomenclature_distance.save(file_name_no_ext + '_nomdist')
        res = {
            'nomenclature_distance': os.path.basename(file_name_no_ext) + '_nomdist',
            'seq_len': self.seq_len
        }
        with open(file_name_no_ext + '.json', 'w') as f:
            json.dump(res, f)
    
    @staticmethod
    def load(bdd: PostGre_SQL_DB, file_name_no_ext):
        """
        Charge la nomenclature_distance depuis file_name_no_ext + '_nomdist',
        puis charge le batcher depuis file_name_no_ext.json

        :param bdd: bdd à utiliser pour charger les divers éléments
        :param file_name_no_ext: path de chargement sans extension
        :returns: AnchorPositivePairsBatch
        """
        with open(file_name_no_ext + '.json') as f:
            res = json.load(f)

        current_dir = os.path.dirname(file_name_no_ext)
        nom_dist = NomenclatureDistance.load(bdd, os.path.join(current_dir, res['nomenclature_distance']))
        appb = AnchorPositivePairsBatch(nom_dist, res['seq_len'])
        return appb


if __name__ == "__main__":

    import sys
    sys.path.append('..')
    from data_import.bdd import PostGre_SQL_DB

    save_file = "trainings/0/batcher"
    batcher = AnchorPositivePairsBatch.load(PostGre_SQL_DB(), save_file)
