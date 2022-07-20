PACKAGE DE TRAINING/OPTIMISATION/RUN DES MODELES DE NOMENCLATURES
=================================================================


## 1. BUT DU PACKAGE

Ce package permet d'entrainer, optimiser et faire tourner des modèles de classification basés sur des nomenclature : Activité (code NAF) et Profession (code Prof).

Les nomenclatures contenant de très nombreux codes, et les données d'entrées étant très parcellaires, les modèles utilisés pour ces problèmes ne sont pas des modèles de classification classiques.

### Principes

Les modèles choisis ont été entraînés sur le principe des **modèles siamois**, qui permettent de projeter des inputs proches mais différents vers un espace commun de représentation, de façon à ce qu'ils soient proches dans cet espace s'ils représentent la même entité, et loin si ce n'est pas le cas. Pour cela, on définit une distance dans l'espace de projection, que l'on apprend au modèle à respecter (**distance learning**).

Lors du training:

1. on prend un batch d'inputs apairés : 1 BI - le noeud de nomenclature associé (description textuelle)
2. on passe d'un côté les BI, de l'autre les noeuds, dans le modèle. On se retrouve donc avec 2 batchs de projections
3. On calcule la distance cible en fonction des inputs (a minima, une diagonale pour que chaque BI puisse se projeter vers son noeud associé)
4. On calcule les distances entre toutes les projections de BI et toutes les projections de nomenclatures
5. On compare la matrice de distance des projections à la matrice cible, et on rétropropage l'erreur
    
Le fonctionnement en batch permet une généralisation de l'apprentissage par triplette, dans lequel on montre au modèle pour chaque input, un input dont il doit être proche, et un dont il doit être loin.

Une fois le modèle appris, et avant de le déployer, on précalcule et pré-stocke les projections de chaque noeud de la nomenclature.

Au runtime:

1. un BI arrive
2. on formatte ses données
3. On calcule la projection par le modèle
4. On calcule la similarité (inverse de la distance) entre cette projection et celles de l'ensemble des noeuds
5. On ordonne les résultats pour trouver les K noeuds les plus similaires
    
Plusieurs étapes peuvent s'ajouter :
    
4bis. On applique un post-process modifiant les similarités sur divers critères
5bis. Si on souhaite coder automatiquement, on applique un critère à définir pour décider si le résultat est suffisamment clair pour décider que le noeud le plus similaire est le bon.
    

### Application aux cas concrets de la NAF et de la profession

#### modèle

Dans le cas qui nous occupe, les données des BI sont parcellaires : 
- elles sont textuelles
- de qualité moyenne (remplies à la main)
- courtes (pas de phrases construites)
- globalement éloignées de la taxonomie des nomenclatures

De ce fait, les modèles à base de cellules temporelles sont peu indiquées (peu de relations temporelles à trouver). Nous sommes donc partis sur une approche basée sur les modèles à attention : les transformers.
L'idée est que le modèle apprends à trouver des relations entre les éléments de l'input sans préjuger d'une relation temporelle.

Le modèle complet est donc:
- on nettoie les données (notamment, on les stemme) et on aggrège plusieurs champs
- on projette l'input dans un dictionnaire
  - soit à base de ngrams
  - soit à base de mots
- on créée des embeddings permettant au modèle de comprendre la complexité des données:
  - un embedding des mots/ngrams (peut être des embeddings pré-entrâinés tels que fasttext)
  - un embedding des champs (chaque mot du champ 1 reçoit l'embedding 1, champ 2 -> embedding 2, etc)
  - un embedding de la position dans le champs (le 1er mot de chaque champ -> embedding 1, etc)
- on le passe dans un bloc "encoding" de transformer
- on passe la somme des encoddings dans le modèle, et on récupère un vecteur de taille N


#### distance

Pour faciliter l'apprentissage, on définit ici une distance un peu plus complexe tirant parti de la structure en arbre des nomenclatures:
- distance = 1 pour une correspondance parfaite entre le BI et le noeud de la nomenclature associé
- distance = c^i pour une correspondance entre le BI et les parents du noeud associé au bi, avec:
  - c : un coeff < 1  (ici 0.5)
  - i : la distance de parenté (2 pour un degré 1, 3 pour un degré 2, etc)
    
ex : pour ['98', '982', '9820', '9820Z']:
        [[1.   , 0.5  , 0.25 , 0.125],
         [0.5  , 1.   , 0.5  , 0.25 ],
         [0.25 , 0.5  , 1.   , 0.5  ],
         [0.125, 0.25 , 0.5  , 1.   ]]
        

#### post-process

Deux post-process ont été mis en place:

1. un post-process basé sur la structure en arbre de la nomenclature:
  - on calcule les similarités entre le BI et les noeuds
  - on y ajoute, pour chaque similarité à un noeud, une partie de la somme des similarités des parents du noeuds corrigés de la valeur utilisée dans la matrice de distance précédente:
  
      sim_k = sim_k + alpha * sum(sim_p * c^i for p in parents_k with dist i to node k)
        
  
2. un post-process basé sur la proximité textuelle entre le champ du BI et la description des noeuds
  - on calcule la représentation en trigrammes
  - on calcule la distance entre les 2 représentations (0 si aucun trigramme commun, 1 si l'un ou l'autre est entièrement contenu dans l'autre)
  - on multiplie la similarité par une partie de ce coefficient:
  
      sim_k = sim_k * (1 + beta * coeff_pt_k)
        
        
Les coefficients alpha et beta sont optimisés par un script après le training.


#### Décision pour codage automatisé



## 2. UTILISATION DU PACKAGE

### Training

L'ensemble du training est paramêtré dans config.yaml, qui est copié dans un dossier de training. Tous les résultats des divers scripts sont sauvegardés dans le dossier.

- on entraine le modèle via ./script_train_model.py
- on calcule les performances top-k via ./script_run_top_k.py
- on optimise les post-process via ./script_optimize_postprocess.py
- si besoin, on entraine un modèle de décision de codage auto via ./script_train_top1_autovalidation_classifier.py

On peut ensuite :

- publier le modèle via ./script_publish_model.py remote_publishing_dir [model_dir]
- calculer et stocker les projections via ./script_project_dbtable.py model_dir input_table output_table
- comparer les résultats avec les résultats MCA/Sicore via script_calculate_compared_perfs.py. Ce dernier script est très dépendant du contexte, il y a donc des variables à remplir dedans.


### Runtime

Les projections de la nomenclature sont calculées et stockées post-training. Il suffit de recharger l'ensemble des produits du training pour pouvoir traiter un (ou un batch) de BI.

L'ensemble des fonctions se trouvent dans script_run_top_k.py.

S'il y a besoin d'autocodage, il faut charger le modèle de décision via script_train_top1_autovalidation_classifier.py


### Description du fichier de config

    '''
    minio:                                                  connexion à minio
      endpoint: http://minio.ouest.innovation.insee.eu     endpoint minio (les identifiants doivent être stocké dans les variables d'env)
      trainings_dir: s3://ssplab/aiee2/nomenclatures         dossier racine minio
      sync: True
    local:
      trainings_dir: trainings                              dossier racine local des trainings
    data:                                                   config data
      force_generation: False                               si True, on recrée les data préprocessée même si elles existent
      postgres_sql: 'SELECT cabbi, rs_x, actet_x, actet_c, profi_x, profs_x, profa_x from rp_final_2017 WHERE actet_c IS NOT NULL'                                        requète de la BDD (toutes les données de train/validation/test en seront tirées)
      limit_nb_doc: null                                    si on veut des trainings plus petits
      nomenclature:
        name: NAF                                           nom de la nomenclature (pour chemin de svg)
        topcode: NAF2_1                                     nom de la nomenclature racine dans la table de la BDD
        node_dist_top_to_first_cat: 2                       distance hiérarchique entre la racine et la première grande catégorisation (1er niveau avec plusieurs noeuds de même niveau). ex: NAF -> 2, PCS -> 1
      use_stemmer: True
      ngrams:
        use: False                                          si True, on utilise des ngrams, sinon des mots
        ngrams_size: 3                                      taille de ngrams
      fasttext:
        use: True                                           si True, on utilise des embeddings préentrainés fasttext
        remote_directory: 'fasttext'                        dossier de stockage sur minio (pour ne pas retélécharger le modèle à chaque fois)
        embedding_size: 210                                 taille d'embeddings à utiliser (de base 300, mais les embeddings peuvent être resizé)
        trainable: True                                     Si true, les embeddings ne sont pas fixés pendant l'entrainement
    trainings:                                              config de training         
      model: TransformerModel                               nom de la classe de modèle
      data:
        gt_column : actet_c                                 colonne de ground truth
        input_columns: ['actet_repr', 'rs_repr', 'prof_repr'] colonne d'inputs (post pré-process)
      model_params:                                         
        seq_len: 20                                         taille max de sequence (calculé automatiquement prétraining)
        embedding_size: 210                                 taille d'embeddings (si on utilise fasttext, sera surchargé par la taille de fasttext)
        nb_blocks: 1                                        nb de bloc transformer
        nb_heads: 3                                         nb de tête d'attention
        ff_dim: 64                                          taille de la fully connected interne des blocs
        dropout: 0.1
        dense_sizes: [128]                                  couches fully connected post dernier bloc transformer. La dernière couche est l'output du modèle
      training_params:
        batch_size: 32                                      taille de batch
        nb_epochs: 20                                       nb d'époques max de training
    post_process:
      top_X_to_optimize: 5                                  critère d'optimisation du postprocess
      nb_trials: 50                                         nb d'essai à réaliser
      optim_timeout_in_min: 2                               timeout de l'optimisation
      alpha_tree_mod: 0                                     parametre alpha (post-process 1) (rempli par le script d'optimisation)
      beta_str_sim_mod: 0                                   parametre beta (post-process 2) (rempli par le script d'optimisation)
    top1_classifier:
      metric_to_maximize: 'precision'                       paramêtre à optimiser pour le codage auto
      scaler_file: ""                                         fichier du scaler (rempli auto par le script)
      classifier_file: ""                                     fichier du modèle (rempli auto par le script)
      threshold: 0.5                                        meilleur threshold trouvé à appliquer
    
    '''

### Structure d'un dossier de training

Structure des dossiers :

    
    |- fasttext
    |    |- cc.fr.XXX.bin : fichier d'embeddings fasttext français taille XXX
    |- trainings
    |    |- NAF
    |    |    |- 2020_11_02_01
    |    |    |    |- config.yaml                                         fichier de config
    |    |    |    |- paths.json                                          path du training sur minio

    |    |    |    |- cabbi_test.csv                                      list des cabbi des BI de test
    |    |    |    |- train_weights                                       dossier des poids svg pendant le training
    |    |    |    |    |- best_model_XX-score
    |    |    |    |    |    |- fichiers de svg tensorflow pour un modèle
    |    |    |    |- training.log                                        log de training
    |    |    |    |- training_history.json                               historique tensorflow du training
    |    |    |    |- batcher.json                                        svg du AnchorPositivePairsBatch du training
    |    |    |    |- batcher_nomdist.json                                svg de la NomenclatureDistance
    |    |    |    |- batcher_nomdist_nomenclature.json                   svg de la Nomenclature
    |    |    |    |- batcher_nomdist_nomenclature_cleaned.json           svg des descriptions cleanées de la nomenclature
    |    |    |    |- batcher_nomdist_nomenclature_emb_dict.pkl           svg du dictionnaire du voc général
    |    |    |    |- batcher_nomdist_nomenclature_hot_encodings.npy      svg de l'encodage en trigrammes de la nomenclature (postprocess)
    |    |    |    |- batcher_nomdist_nomenclature_hot_encodings_voc.pkl  svg du vocabulaire trigramme de la nomenclature (postprocess)
    |    |    |    |- batcher_nomdist_nomenclature_projs.npy              svg des projections de la nomencalture

    |    |    |    |- top_k.json                                          résultat des top-k
    |    |    |    |- test_results.csv                                    résultats des top-k par BI
    |    |    |    |- topk.log                                            log de run_top_k.py

    |    |    |    |- postprocess_optim                                   dossier de résultats intermédiaires de l'optim
    |    |    |    |    |- optimisation_results.json                      résultat de l'optim et des essais
    |    |    |    |    |- top_k_i.json                                   top-k pour l'essai i
    |    |    |    |    |- test_results_i.csv                             top-k apar BI de train pour l'essai i
    |    |    |    |- optim_top_k.json                                    résultat de l'optim
    |    |    |    |- optim_test_cabbis.csv                               cabbis des BI de test optimisation
    |    |    |    |- optim_test_results.csv                              résultat de l'optim par BI pour les docs de test
    |    |    |    |- optim_train_cabbis.csv                              cabbis des BI de train optimisation
    |    |    |    |- optim_train_results.csv                             résultat de l'optim par BI pour les docs de train de l'optim

    |    |    |    |- top1_scaler.pkl                                     fichier du scaler d'auto-encodage
    |    |    |    |- top1_classifier.pkl                                 fichier du modèle d'auto-encodage
    |    |    |    |- top1_training.json                                  résultat du training

    |    |    |- data(\_stemmed)(\_ngrams)(\_fasttext_XXX)                dossier de data préprocessé
    |    |    |    |- NAF2_1_sql_request.yaml                             requète SQL pour contrôle
    |    |    |    |- NAF2_1_data.csv                                     data cleanée
    |    |    |    |- NAF2_1_data_dict.pkl                                dictionnaire (ngram ou mot)
    |    |    |    |- NAF2_1_data_dict_ft_vectors.npy                     embeddings FT si on utilise FT
    |    |    |    |- NAF2_1_data_prepared.csv                            data préprocessée
    |    |    |    |- NAF2_1_nomdistance.json                             svg de la NomenclatureDistance
    |    |    |    |- NAF2_1_nomdistance_nomenclature.json                svg de la Nomenclature
    |    |    |    |- NAF2_1_nomdistance_nomenclature_cleaned.json        svg des descriptions cleanées de la nomenclature 
    |    |    |    |- NAF2_1_nomdistance_nomenclature_emb_dict.pkl        svg du dictionnaire du voc général
    |    |    |    |- NAF2_1_nomdistance_nomenclature_emb_dict_ft_vectors.npy  svg des embeddings FT


## 3. STRUCTURE DU PACKAGE

- **requirements.txt**:                            fichier requirements python
- **config.yaml**:                                 fichier de config (voir plus haut)
- **logging.conf.yaml**:                           fichier de config de log
- **README.md**:                                   readme (ce fichier)

- **training_classes**:                            classes de code
    - **naf_cleaner.py**:                          classe de préprocess des données 
    - **preprocessing.py**:                        fonctions de préprocess
    - **embedding_dictionary.py**:                 classe de vocabulaire pour le préprocess
  
    - **fasttext_singleton.py**:                   classe de gestion de fasttext

    - **nomenclature.py**:                         classe de gestion des nomenclatures
    - **nomenclature_distance.py**:                classe de distance associée à la nomenclature

    - **similarity_model.py**:                     classe de modèle siamois / distance learning
    - **token_field_and_position_embedding.py**:   classe d'embeddings voc/champs/position
    - **anchor_positive_pairs_batch.py**:          classe de batching par paires
    - **training_model.py**:                       classe mère de modèle d'entrainement
    
    - **lstm_model.py**:                           TrainingModel basé sur des lstm
    - **transformer_model.py**:                    TrainingModel basé sur les transformers

- **script_train_model.py**:                              script de training (exécutable)
- **script_run_top_k.py**:                                script de calcul des top-k (exécutable)
- **script_optimize_postprocess.py**:                     script d'optimisation de post-process (exécutable)
- **script_train_top1_autovalidation_classifier.py**:     script de training pour auto-encodage (exécutable)
- **script_publish_model.py**                             script de publication d'un modèle
- **script_project_dbtable.py**                           script de projection d'une table de BI
- **script_calculate_compared_perfs.py**                  script de comparaison entre les nouveaux résultats et les résultats MCA/Sicore
- **training_utils.py**:                           fonctions utilitaires pour les scripts précédents

- **fasttext**:                                    voir plus haut
- **trainings **:                                  voir plus haut