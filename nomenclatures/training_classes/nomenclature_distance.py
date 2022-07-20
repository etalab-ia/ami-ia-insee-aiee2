import tensorflow as tf
import numpy as np
import json

import os, sys
sys.path.append(os.path.join('..', '..'))
from data_import.bdd import PostGre_SQL_DB
from .nomenclature import Nomenclature


class NomenclatureDistance:
    def __init__(self, nomenclature: Nomenclature, decrease_rate=0.5):
        """
        Classe permettant de calculer les distances à appliquer lors du training
        entre les noeuds de la nomenclature, et de calculer la matrice de 
        distance pour un batch de training donné

        La distance appliquée est :
            - 1 si codes égaux
            - decrease_rate si l'un est le noeud-parent de l'autre
            - decrease_rate^2 si l'un est le noeud-parent du noeud-parent de l'autre
            - etc


        :param nomenclature: Nomenclature à traiter
        :param decrease_rate: taux de décroissance de la distance
        """
        self.nomenclature = nomenclature
        # matrice encodant la dépendance d'un noeud à ses parents
        self._scores_mod_mat = None
        # matrice de distance complète.
        # vaut _scores_mod_mat + _scores_mod_mat.T + eye
        self._distance_mat = None
        self._distance_mat_tf = None
        # index des noeuds pour les matrice
        self.nodes_index = None
        self.decrease_rate = decrease_rate
        self._build_distance()
        
    def _build_distance(self):
        """ Crée la matrice de distance de la nomenclature """
        self.nodes_index = {n: i for i, n in enumerate(self.nomenclature.nodes.keys())}
        self._distance_mat = np.zeros([len(self.nodes_index), len(self.nodes_index)])
        
        def _get_children_at_k(node_name, k):
            """ récupère tous les noeuds fils du k-ieme niveau """
            if k == 0 : return []
            if k == 1 : return [n.name for n in self.nomenclature.get_node(node_name).children]
            return sum([_get_children_at_k(n.name, k-1) for n in self.nomenclature.get_node(node_name).children], [])
            
        def _setup_node_distances(node_name, list_of_parents_from_furthest):
            """ calcule la distance entre les noeuds au sens parentèle """
            current_dist = 1
            node_idx = self.nodes_index[node_name]
            for p in reversed(list_of_parents_from_furthest):
                current_dist *= self.decrease_rate
#                 print(node_idx, self._nodes_index[p], current_dist)
                self._distance_mat[node_idx, self.nodes_index[p]] = current_dist
            
        def _propagate_node_distances(node_label, list_of_parents_from_furthest):
            """ propagation de la distance dans la hiérarchie """
            children = [n.name for n in self.nomenclature.get_node(node_label).children ]
            if not len(children):
                _setup_node_distances(node_label, list_of_parents_from_furthest)
            else:
                _setup_node_distances(node_label, list_of_parents_from_furthest)
                [_propagate_node_distances(child, list_of_parents_from_furthest + [node_label]) for child in children]
        
        start_nodes = _get_children_at_k(self.nomenclature.name, self.nomenclature.node_dist_top_to_first_cat)
        for node_name in start_nodes:
            _propagate_node_distances(node_name, [])
                
        self._scores_mod_mat = self._distance_mat.copy()
        self._distance_mat += self._distance_mat.T
        self._distance_mat += np.eye(len(self.nodes_index))
        self._distance_mat_tf = tf.convert_to_tensor(self._distance_mat)
    
    def get_distance_mat(self, list_of_nodes=None):
        """
        récupérer la matrice de distance pour une liste d'id de noeuds 

        :param list_of_nodes: liste de str. si None, on récupère toute la matrice
        :returns: np.array de dim (len(list_of_nodes), len(list_of_nodes))
        """
        if list_of_nodes is None:
            return self._distance_mat
        nodes_ind = [self.nodes_index[n] for n in list_of_nodes]
        return self._distance_mat[nodes_ind, :][:,nodes_ind]
    
    def get_distance_mat_from_tf_indices(self, list_of_node_indices):
        """
        Pareil que get_distance_mat, mais pour des tensor Tensorflow
        """
        return tf.gather(tf.gather(self._distance_mat_tf, list_of_node_indices, axis=0), list_of_node_indices, axis=1)
    
    def get_score_modifier_mat(self, list_of_nodes=None):
        """
        récupérer la matrice de dépendance parentale pour une liste d'id de noeuds
        
        :param list_of_nodes: liste de str. si None, on récupère toute la matrice
        :returns: np.array de dim (len(list_of_nodes), len(list_of_nodes))
        """
        if list_of_nodes is None:
            return self._scores_mod_mat
        nodes_ind = [self.nodes_index[n] for n in list_of_nodes]
        return self._scores_mod_mat[nodes_ind, :]
    
    def save(self, file_name_no_ext):
        """
        Sauvegarde la Nomenclature dans file_name_no_ext_nomenclature,
        puis self dans file_name_no_ext.json

        :param file_name_no_ext: chemin ou sauvegarder, sans ext
        """
        self.nomenclature.save(file_name_no_ext + '_nomenclature')
        res = {
            'nomenclature': os.path.basename(file_name_no_ext) + '_nomenclature',
            'decrease_rate': self.decrease_rate
        }
        with open(file_name_no_ext + '.json', 'w') as f:
            json.dump(res, f)

    @staticmethod
    def load(bdd: PostGre_SQL_DB, file_name_no_ext):
        """
        Charge la Nomenclature depuis file_name_no_ext_nomenclature,
        puis self depuis file_name_no_ext.json

        :param bdd: driver postgres
        :param file_name_no_ext: chemin depuis où charger, sans ext
        :returns: NomenclatureDistance
        """
        with open(file_name_no_ext + '.json') as f:
            res = json.load(f)

        current_dir = os.path.dirname(file_name_no_ext)
        nom = Nomenclature.load(bdd, os.path.join(current_dir, res['nomenclature']))
        nom_dist = NomenclatureDistance(nom, res['decrease_rate'])
        return nom_dist


if __name__ == '__main__':
    
    bdd = PostGre_SQL_DB()
    naf = Nomenclature(bdd, 'NAF2_1')
    naf_distances = NomenclatureDistance(naf)
    print(naf_distances.get_distance_mat(["01", "77", "771", "7712", "7712Z", "0116", "0119"]))
    naf_distances.get_distance_mat_from_tf_indices(
        tf.convert_to_tensor([naf_distances.nodes_index[n] for n in ["01", "77", "771", "7712", "7712Z"]]))
