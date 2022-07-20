from anytree import Node, RenderTree
import numpy as np
import json
from collections import Counter

import os, sys
sys.path.append(os.path.join('..', '..'))
from data_import.bdd import PostGre_SQL_DB
from .embedding_dictionary import EmbeddingDictionary


class Nomenclature(Node):
    
    def __init__(self, bdd: PostGre_SQL_DB, code, node_dist_top_to_first_cat=2,
                 parent=None):
        """
        Classe permettant de charger une nomenclature depuis la DB, et de la représenter sous forme 
        d'arbre
        (basé sur la lib anytree : https://anytree.readthedocs.io/en/2.8.0/api/anytree.node.html)
        
        la Nomenclature est le noeud racine, et utilise la table bdd.Nomenclatures
        valeurs accessibles : 
            - id : nct_id de la nomenclature dans bdd.nomenclatures
            - name: code de la nomenclature
            - parent : a priori None
            - desc: libelle de la nomenclature
            - desc_cleaned : libelle nettoyé
            - bdd : PostGre_SQL_DB à utiliser
            - nodes : dictionnaire {name: Node} de l'ensemble des noeuds de l'arbre
        
        les noeuds intermediaires et les feuilles sont des modalites de la nomenclature racine, ou de nomenclatures qui dépendent de la racine (via Nomenclature.nct_agr)
        valeurs accessibles:
            - id : code de la modalite dans bdd.Modalites
            - name: code de la modalite
            - nct_id: nct_id de la modalite
            - desc : libelle de la modalite
            - desc_cleaned : libelle nettoyé
        pour des raisons de volume de donnée, l'aide de la modalité n'est pas récupérée, mais est accessible via get_node_help
        
        La classe permet ensuite de calculer et manipuler diverses représentations des noeuds,
        ainsi que de calculer les top-k pour un batch de projections donné
        
        :param bdd: PostGre_SQL_DB
        :param code: code de la nomenclature à récupérer
        :param node_dist_top_to_first_cat: distance hiérarchique entre la racine et la première grande
                                           catégorisation (1er niveau avec plusieurs noeuds de même niveau)
        :param parent: si on ne crée pas la racine
            
        """
        self.nomenclature_code = code
        self.node_dist_top_to_first_cat = node_dist_top_to_first_cat
        self.bdd = bdd
        self.nodes = {}
        self.root_node= None
        
        # récupération de la nomenclature
        bdd_node = bdd.read_from_sql(F"SELECT * FROM nomenclatures where code = '{code}' ").to_numpy()[0]
        super().__init__(bdd_node[3], parent=parent, id=bdd_node[0], desc=bdd_node[1], desc_cleaned=None)
        self.nodes[bdd_node[3]] = self
        self.root_node = self
        
        # recupération de ses modalités
        children = bdd.read_from_sql(F"SELECT * FROM modalites where nct_id = {self.id} ").to_numpy()
        for child in children:
            node = Node(child[0], parent=self, id=child[0], desc=child[1], nct_id=child[4], desc_cleaned=None) #, help=child[3])
            if child[0] == self.name:
                self.root_node = node
            self.nodes[child[0]] = node
            
        # récupération des sous-nomenclatures
        sub_nom = bdd.read_from_sql(F"SELECT * FROM nomenclatures where nct_agr = '{code}' ").to_numpy()
        has_subnom = len(sub_nom)
        while has_subnom:
            sub_nom = sub_nom[0]
            children = bdd.read_from_sql(F"SELECT * FROM modalites where nct_id = {sub_nom[0]} ").to_numpy()
            for child in children:
                assert child[2] in self.nodes
                node = Node(child[0], parent=self.nodes[child[2]], id=child[0], desc=child[1], nct_id=child[4], desc_cleaned=None) #, help=child[3])
                if self.nodes[child[2]].name == node.name \
                    and self.nodes[child[2]].id == node.id \
                        and self.nodes[child[2]].desc == node.desc:
                    node.parent = self.nodes[child[2]].parent
                    node.parent.children = [c for c in node.parent.children if c != self.nodes[child[2]] ]
                    del self.nodes[child[2]]
                self.nodes[child[0]] = node
                
            sub_nom = bdd.read_from_sql(F"SELECT * FROM nomenclatures where nct_agr = '{sub_nom[3]}' ").to_numpy()
            has_subnom = len(sub_nom)
            
        # dictionnaire pour calculer la représentation projetable des noeuds (dans desc_clean)
        self.embeddings_dict = None
        # projections des noeuds pour un modèle donné
        self.projections_reverse_ind = None
        self.projections = None
        # dernière similarités calculées
        self.last_similarity_values = None
        # dictionnaire pour calculer la proximité ngram de desc_clean avec une string
        self.ngram_voc = None
        # hot encodings de la version ngram de desc_clean
        self.ngram_hot_encodings = None
            
    def get_node(self, node_name):
        """
        récupérer un noeud de l'arbre
        
        :param node_name: code de la modalite à récupérer
        :returns: Node
        """
        return self.nodes[node_name]
        
    def get_node_help(self, node_name):
        """
        Récupérer l'aide d'un noeud
        
        :param node_name: code de la modalite à récupérer
        :returns: str
        """
        node = self.nodes[node_name]
        return bdd.read_from_sql(F"SELECT * FROM modalites where nct_id = {node.nct_id} and code = '{node.id}' ").to_numpy()[0][3]
        
    def display(self):
        """
        imprimer la nomenclature
        """
        for pre, _, node in RenderTree(self):
            print("%s%s:%s" % (pre, node.name, node.desc))
            
    def display_node(self, node_name):
        """
        imprimer le sous arbre dont la racine est :param node_name:
        """
        for pre, _, node in RenderTree(self.get_node(node_name)):
            print("%s%s:%s" % (pre, node.name, node.desc))

    def clean_node(self, node_name, clean_func):
        """
        nettoie la description d'un noeud avec clean_func et la stocke dans desc_cleaned
        
        :param node_name: noeud à traiter
        :param clean_func: fonction à appliquer
        :returns: str
        """
        n = self.get_node(node_name)
        n.desc_cleaned = clean_func(n.desc)
        return n.desc_cleaned

    def build_nomenclature_embeddings(self, emb_dict: EmbeddingDictionary):
        """
        calculer la représentation embeddings de chaque node
        
        :param emb_dict: dictionnaire à utiliser
        """
        self.embeddings_dict = emb_dict
        for _, v in self.nodes.items():
            if not v.desc_cleaned:
                raise RuntimeError('nomenclature not cleaned')
            v.embedding_repr = self.embeddings_dict.convert_voc(v.desc_cleaned)
            
    def get_nomenclature_embeddings(self, node_name):
        """
        récupérer la représentation embeddings d'un node
        
        :param node_name: noeud à traiter
        :returns: list of float
        """
        if self.embeddings_dict is None:
            raise RuntimeError('Nomenclature embeddings not calculated')
        node = self.nodes[node_name]
        return node.embedding_repr
    
    def build_nomenclature_projections(self, embeddings_projection_func):
        """
        calculer les projections des noeuds pour le modèle donnée
        
        :param embeddings_projection_func: fonction de projection (vient du modèle entrainé)
        """
        projections = []
        self.projections_reverse_ind = {}
        for i, (k, v) in enumerate(self.nodes.items()):
            self.projections_reverse_ind[i] = k
            projections.append(embeddings_projection_func(v.embedding_repr))
        self.projections = np.stack(projections, axis=0)
        
    def get_topk(self, bi_projections, similarity_func, nb_top_values=5, 
                 alpha_tree=0, nom_distance=None,
                 beta_str_sim=0, cleaned_inputs=None):
        """
        Calcul des top-k similarités pour un batch de projections de BI
        Plusieurs corrections peuvent être appliquées:
            - correction via chaine de parentèle : la similarité de chaque noeud se voit 
            ajouter alpha_tree * somme des (similarités de ses parents * coeff de distance)
            - correction de similarité textuelle : la similarité de chaque noeud se voit ajouter
            beta_str_sim * (nb de ngrams communs entre l'input et le noeud.desc_clean)
            
        :param bi_projections: projection ou batch de projections
        :param similarity_func: fonction de similarité utilisé pendant le training
        :param nb_top_values: int. nb de valeurs à renvoyer
        :param alpha_tree: float. Paramètre de correction de chaine de parentèle
        :param nom_distance: nomenclature_distance permettant de récupérer les parents et les coeffs associés
        :param beta_str_sim: float. Paramètre de correction de similarité textuelle
        :param cleaned_inputs: texte de l'input (ou liste si batch)
        :return: [[noms de noeuds], [scores de similarité]] (ou liste si batch) 
                    - trié par similarité décroissante
                    - de taille nb_top_values
        """
        single_mode = False
        if not isinstance(bi_projections, list):
            single_mode = True
            bi_projections = [bi_projections]
            cleaned_inputs = [cleaned_inputs]
        results = []
        self.last_similarity_values = []
        for i, bi_projection in enumerate(bi_projections):
            base_similarities = similarity_func(np.expand_dims(bi_projection, axis=0), self.projections).numpy()[0]
            similarities = base_similarities.copy()
            if alpha_tree:
                similarities += alpha_tree * self.get_tree_score_modifiers(nom_distance, base_similarities)
            if beta_str_sim:
                similarities += beta_str_sim * self.get_trigram_dist_modifiers(cleaned_inputs[i], base_similarities)
            self.last_similarity_values.append(similarities)
            top_idx = (-similarities).argsort()[:nb_top_values]
            top_codes = [self.nodes[self.projections_reverse_ind[i]].id for i in top_idx]
            top_similarities = [similarities[i] for i in top_idx]
            results.append([top_codes, top_similarities])
        if single_mode:
            return results[0]
        return results
    
    def get_tree_score_modifiers(self, nom_distance, similarity_scores):
        """
        Renvoie les modifications de score de similarité par chaine de parentèle
        
        :param nom_distance: nomenclature_distance à appeler
        :param similarity_scores: scores de similarité de base
        :return: list[float]
        """
        return np.matmul(nom_distance.get_score_modifier_mat(), similarity_scores)
    
    def _project_to_ngram_hot(self, text):
        """
        projette un input en ngram et l'encode en one-hot
        
        :param text: texte à encoder
        :return: list[int]
        """
        if self.ngram_voc is None:
            raise RuntimeError('please call create_trigram_repr before')
        nb_ngrams = self.ngram_voc.get_vocab_size()
        trigram_inds = self.ngram_voc.convert_voc(text)
        counts = Counter(trigram_inds)
        hot_encode = np.zeros((nb_ngrams))
        hot_encode[list(counts.keys())] = list(counts.values())
        return hot_encode
    
    def create_trigram_repr(self, size_ngrams=3):
        """
        crée un dictionnaire interne de ngram sur les desc_cleaned
        
        :param size_ngrams: taille de ngram à appliquer
        """
        embedding_voc_class = EmbeddingDictionary.factory(use_ngrams=True)
        self.ngram_voc = embedding_voc_class(pad_token_at_start=True, 
                                             distinct_null_token=True,
                                             ngrams_size=size_ngrams)
        self.ngram_voc.add_data([n.desc_cleaned for n in self.nodes.values()])
        self.ngram_voc.finalize_dict()
        
        hot_encodings = []
        for _, v in list(self.nodes.items()):
            if not v.desc_cleaned:
                raise RuntimeError('nomenclature not cleaned')
            hot_encodings.append(self._project_to_ngram_hot(v.desc_cleaned))
        self.ngram_hot_encodings = np.stack(hot_encodings, axis=0)
        
    def get_trigram_dist_modifiers(self, input_cleaned_text, similarity_scores):
        """
        Renvoie les modifications de score de similarité textuelle
        
        :param input_cleaned_text: texte du BI
        :param similarity_scores: scores de similarité de base
        :return: list[float]
        """
        if self.ngram_voc is None:
            raise RuntimeError('please call create_trigram_repr before')
            
        input_ngrams_hot = self._project_to_ngram_hot(input_cleaned_text)
        ngram_sim = 0
        max_sim = sum(input_ngrams_hot)
        if max_sim != 0:
            ngram_sim = np.sum(np.minimum(self.ngram_hot_encodings, input_ngrams_hot), axis=1) / max_sim
        return np.multiply(ngram_sim, similarity_scores)

    def get_last_nodes_scores(self, node_names):
        """
        Renvoie le dernier score de similarité calculé pour les noeuds
        
        :param node_names: noeud à traiter
        :return: list[float]
        """
        if self.last_similarity_values is None:
            raise RuntimeError()

        if not isinstance(node_names, list):
            node_names = [node_names]
        nodes_inds = {}
        for k, v in self.projections_reverse_ind.items():
            if v in node_names:
                nodes_inds[v] = k
        nodes_proj_inds = [nodes_inds[n] for n in node_names]
        res = zip(*[proj[nodes_proj_inds].tolist() for proj in self.last_similarity_values])
        return list([list(r) for r in res])
    
    def save(self, file_name_no_ext):
        """
        Sauvegarde des différents éléments
        
        fichiers créés : 
            - file_name_no_ext + '_cleaned.json' : ensemble des desc_cleaned
            - file_name_no_ext + '_projs.npy' : ensemble des projections
            - file_name_no_ext + '_hot_encodings.npy': ensemble des ngram hot encodings
            - file_name_no_ext + '.json': description de la nomenclature
        De plus, les 2 EmbeddingsVoc sont sauvegardé avec comme bases respectives 
            - file_name_no_ext + '_emb_dict'
            - file_name_no_ext + '_hot_encodings_voc'
        
        :param file_name_no_ext: base de nommage
        :return None:
        """
        res = {'nomenclature_code': self.nomenclature_code,
               'node_dist_top_to_first_cat': self.node_dist_top_to_first_cat}
        if self.root_node.desc_cleaned:
            cleaned_values = {k: n.desc_cleaned for k, n in self.nodes.items()}
            with open(file_name_no_ext + '_cleaned.json', 'w') as f:
                json.dump(cleaned_values, f)
            res['cleaned_values'] = os.path.basename(file_name_no_ext) + '_cleaned.json'
        if self.embeddings_dict is not None:
            self.embeddings_dict.save(file_name_no_ext + '_emb_dict')
            res['embeddings_dict_class'] = self.embeddings_dict.__class__.__name__
            res['embeddings_dict'] = os.path.basename(file_name_no_ext) + '_emb_dict'
        if self.projections is not None:
            np.save(file_name_no_ext + '_projs.npy', self.projections)
            res['projections'] = os.path.basename(file_name_no_ext) + '_projs.npy'
            res['projections_reverse_ind'] = self.projections_reverse_ind
        if self.ngram_voc is not None:
            self.ngram_voc.save(file_name_no_ext + '_hot_encodings_voc')
            res['hot_encodings_voc'] = os.path.basename(file_name_no_ext) + '_hot_encodings_voc'
            np.save(file_name_no_ext + '_hot_encodings.npy', self.ngram_hot_encodings)
            res['hot_encodings'] = os.path.basename(file_name_no_ext) + '_hot_encodings.npy'
        with open(file_name_no_ext + '.json', 'w') as f:
            json.dump(res, f)

    @staticmethod
    def load(bdd: PostGre_SQL_DB, file_name_no_ext):
        """
        Chargement d'une nomenclature sauvegardée
        
        :param bdd: PostGre_SQL_DB à utiliser
        :param file_name_no_ext: base de nommage utilisée à la sauvegarde
        :return: une instance de Nomenclature
        """
        with open(file_name_no_ext + '.json') as f:
            res = json.load(f)

        current_dir = os.path.dirname(file_name_no_ext)
        nom = Nomenclature(bdd, 
                           res['nomenclature_code'],
                           res.get('node_dist_top_to_first_cat', 2))  # 2 -> backward compatibility
        if 'cleaned_values' in res:
            with open(os.path.join(current_dir, res['cleaned_values'])) as f:
                cleaned_values = json.load(f)
            for k, n in nom.nodes.items():
                n.desc_cleaned = cleaned_values[k]
        if 'embeddings_dict' in res:
            embeddings_dict = EmbeddingDictionary.load(os.path.join(current_dir, res['embeddings_dict']))
            nom.build_nomenclature_embeddings(embeddings_dict)
        if 'projections' in res:
            nom.projections = np.load(os.path.join(current_dir, res['projections']))
            nom.projections_reverse_ind = {int(k): v for k, v in res['projections_reverse_ind'].items()}
        if 'hot_encodings_voc' in res:
            nom.ngram_voc = EmbeddingDictionary.load(os.path.join(current_dir, res['hot_encodings_voc']))
            nom.ngram_hot_encodings = np.load(os.path.join(current_dir, res['hot_encodings']))
        return nom        


if __name__ == "__main__":
    bdd = PostGre_SQL_DB()
    
    insee = Nomenclature(bdd, 'DR_INSEE')
    insee.display()
    
    naf = Nomenclature(bdd, 'PCS1')
#     naf.display()
    
    naf.display_node('012')