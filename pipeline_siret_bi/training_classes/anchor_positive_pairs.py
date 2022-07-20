import numpy as np
from tensorflow import keras


def pad_seq(seq, seq_len):
    pad_token = 0
    return seq + [pad_token for _ in range(seq_len - len(seq))]


class AnchorPositivePairs(keras.utils.Sequence):

    def __init__(self, num_batchs, batch_size, input_data, pos_data, labels, seq_len):
        """
        Générateur de paire bi/sirus.   
        Les input sont paddé, on génère aussi l'emplacement des colonne et des mots dans les colonnes.
    
        Ex:
            Input: [0,1,2,3,4,5,6,7,8,9,10 ...] de taille variable

            Output -> [0,1,2,3,4,5,6,8,9,10,0,0, ... n] n de taille fixe
                    [0,0,0,1,1,1,2,2,2,3,3,3 ... n] int représentant l'origine de la colonne
                    [1,2,3,1,2,3,1,2,3,1,2,3 ... ,] int représentant la position du mot dans la colonne (1er, 2ème ...)
                    
                    Le triple padding est fait pour les BI et sirus, donc 6 padding en sortie.
                    On conserve le siret pour l'organisation de la matrice d'aprentissage donc 7  dimension en sortie.
        
        :param num_batchs: nombre de batch à créer (le générateur s'arrête après)
        :param batch_size: taille des batchs en exemples
        :param input_data: np.array N*F1*?. data input (représentation BI). N BI, F1 champs, ? token / champ
        :param pos_data: np.array N*F2*?. data positives (représentation Entreprise)
        :param labels: np.array N. Labels (Siret)
        :param seq_len: longueur d'une séquence (pour le padding)
        """
        self.num_batchs = num_batchs
        self.input_data = input_data
        self.current_pos = 0
        self.pos_dat = pos_data
        self.batch_size = batch_size
        self.labels = labels
        self.seq_len = seq_len

    def __len__(self):
        return self.num_batchs
    
    @staticmethod
    def get_fields(seq):
        """
        Renvoie l'indice de champs pour chaque token de la séquence

        :param seq: list of list of tokens (tokens par champs dans un enregistrement)
        """
        fields = []
        for index, item in enumerate(seq):
            fields += [index + 1 for _ in item]
        return fields
    
    @staticmethod
    def get_positions(fields):
        """
        Renvoie l'indice de position dans le champs pour chaque token de la séquence

        :param seq: list of list of tokens (tokens par champs dans un enregistrement)
        """
        p = 1
        cf = fields[0]
        positions = []
        for f in fields:
            if f != cf:
                cf = f
                p = 1
            positions.append(p)
            p += 1
        return positions

    def __getitem__(self, _idx):
        """
        Création du batch lui-même. L'output est N*7:
            - 0 -> 2 : représentation du BI
            - 3 -> 5 : représentation de l'entreprise
            - 6 : siret clé
        """
        flatten = lambda l: [item for sublist in l for item in sublist]

            
        x = np.empty((7, self.batch_size, self.seq_len), dtype = np.float32)
        for i in range(self.batch_size):
            
            list_item_anchor = []
            for item in self.input_data[self.current_pos]:
                list_item_anchor.append(item)
            x[0, i] = pad_seq(flatten(list_item_anchor), self.seq_len)
            fields = self.get_fields(list_item_anchor)
            x[1, i] = pad_seq(fields, self.seq_len)
            x[2, i] = pad_seq(self.get_positions(fields), self.seq_len)
            
            list_positive_item = []
            for item in self.pos_dat[self.current_pos]:
                list_positive_item.append(item)
            x[3, i] = pad_seq(flatten(list_positive_item), self.seq_len)
            positive_fields = self.get_fields(list_positive_item)
            x[4, i] = pad_seq(positive_fields, self.seq_len)
            x[5, i] = pad_seq(self.get_positions(positive_fields), self.seq_len)
            
            x[6, i] = pad_seq(self.labels[self.current_pos], self.seq_len)
            
            self.current_pos += 1
            self.current_pos %= len(self.input_data)
        return x
    
