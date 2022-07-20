# PACKAGE PIPELINE_SIRET_BI
===========================


Ce package contient les classes et script pour entraîner et appliquer un modèle de codage de SIRET.


## Contenu
----------

### Fichiers de configuration
-----------------------------

- *config.yaml* : fichier contenant toutes les opération DataEng/ML à lancer pour le training
- *config_si.yaml*: config utilisée pour la prédiction sur un input de type Sirus
- *config_bi.yaml*: config utilisée pour la prédiction sur un input de type BI

- *logging.conf.yaml*: fichier de config pour le log
- *launch* : fichier pour lancer plusieurs fois le pipeline

### Dossiers
------------

- *elastic* : dossier qui contient toutes les requêtes ES
- *geocodage*: dossier contenant les données utilisées par le géocodage
- *training_classes*: classes de cleaning/processing/training :
    - *CleanerActetX.py* : fichier contenant un cleaning de l'actetX
    - *CleanerAdrCltX.py* : fichier contenant un cleaning de CltX pour sirus
    - *CleanerAdrDltX.py* : fichier contenant un cleaning de DltX pour sirus
    - *CleanerConcat_profs.py* : fichier contenant un cleaning qui réunit les champs prof*_x
    - *CleanerRemoveSpecialChar.py* : fichier contenant un cleaning sur les char spec
    - *Model.py* : fichier contenant la classe mère des Modèles
    - *ModelSimilarity.py* : fichier contenant le mdoèle de similarité BI/SIRUS
    - *ProcessMLPSiamese.py* : fichier contenant le modèle siamois
    - *Process.py* : fichier contenant la classe mère des Process

    - *preprocessing.py*: fonctions de nettoyage de chaines de caractères
    - *utils.py*: fichier de fonction utilitaire (gestion des fichier en local ou sur minio etc..)


### Fichiers de fonctions
-------------------------

- *geocodage.py* : fonction de géocodage via api ban/bano/poi
- *elastic.py*: driver elastic permettant de charger des données dans ES et de le requêter


### Fichiers de script
----------------------

- *script_pipeline.py* : fichier qui gère l'orchestration de l'execution d'un fichier de config
- *script_runtime.py* : fichier qui contient toutes les fonctions nécessaires au runtime pour charger un model entraîné, calculer les projections via le modèle, gérer les enrichissements divers, géocodage, récupération des prédictions naf, appels ES, etc. Il permet aussi de générer les données nécessaires à l'entrainement du méta-modèle, et à calculer les métriques pré et post meta-modèle
- *script_metamodel_optimisation.py*: optimisation du méta-modèle et du modèle de codage automatique
- *script_publish_model.py*: script permettant de publier un modèle entraîné sur minio pour qu'il soit utilisable dans le service de prédiction

- *script_project_dbtable.py* : fichier permettant de calculer les projections d'une table sql
- *script_export_to_es_projections.py* : fichier d'export des projections vers ES
- *script_export_to_es_bi_enriched_data.py* : fichier qui récupère les données rp geocodées sur minio, les joint aux données postgres et les exporte sur ES
- *script_export_to_es_sirus_enriched_data.py* : fichier qui récupère les données sirus geocodées sur minio, les joint aux données postgres et les exporte sur ES
- *script_generate_groundtruth_file.py* : fichier qui génère les fichiers de grond truth nécessaire à l'entrainement du meta-model (eval) et au calcul des perfs finales (test)
- *script_generate_scoring.py*: script qui génère les fichiers et graphs de métriques end-to-end


## Dossiers créés pendant le training et le runtime
---------------------------------------------------

Lors du training, un dossier est créé par training, dans trainings.
Au runtime, le dossier où écrire tous les fichiers est configurable.


## Autres logiciels
-------------------

Le projet se repose sur une base de donnée postgresql et un clusteur ElasticSearch.

### postgresql
--------------

Les tables postgresql sont :
- *bi_lot* : contient les informations de lot/vague des BI
- *rp_final*: jeu de donnée BI
- *sirus*: jeu de donnée sirus
- *sirus_projection*: projection de la table sirus via le modèle

Afin d'accélérer certains process des index ont été créer:

- Un index sur les cabbi 
- Un index sur le siret (sirus_id + nic)

sur toutes les tables ayant une des colonnes

### ElasticSearch
-----------------

Les index ElasticSearch sont:

- *sirus_e* : donnée enrichie (géocodage) sirus
- *rp_e* : donnée enrichie (géocodage) bi

## Fonctionnement général
-------------------------

Un fichier de configuration contient les opérations et leurs paramètres à éxécuter. Une opération est de type:
    - cleaner: nettoyage de données
    - Process: Opération qui va modifier les données, peut avoir une modèle s'apparente à la phase de feature engineering
    - Modèle: Modèle de machine learning et métriques

Le fichier script_pipeline se charge de l'orchestration de ces opérations.

La sauvegarde sur minio est réalisé ainsi : Chaque run de pipeline crée un dossier unique correspond au jour du début du run et du nombre de run ayant été réalisé ce jour-là. Tout les fichiers du run seront estampillé de cet id, dans ce dossier. L'ID est donc unique.

Le pipeline peut être executé de 3 façon différente: 
-  Uniquement avec les opération de cleaner et de transformer, les fichiers sont sauvegarder dans un dossier commun ("based") et sont destiné à être réutiliser. Attention tout les fichiers déjà présent dans le dossier based seront supprimé.  
- Uniquement avec les opération d'échantillonage et modèle. Dans ce cas les données sont chargé depuis le fichier commun ("based") et les modèles sauvegardé dans leur dossier unique (paragraphe de dessus).
- 2en1 (par défaut), le pipeline est éxécuter du début jusqu'a la fin. (rien n'est sauvegarder dans based)

Les modèles qui peuvent être tunés le sont via hyperot (optimisation bayesien).

Les fichiers les plus importants pour la réconciliation BI/Sirus sont: processMLPSiamese.py et ModelSimilarity.

Lors d'un entrainement, des fichiers "output" sont générés dans le dossier de training:
 - args.txt : fichier contenant les arguments utilisés pour lancer script_pipeline.py
 - paths.json: chemin vers le training sur minio
 - config.yaml: config de training
 - *cabbi_test.p*: fichier pickle des id rp de test (peut être vide, dans ce cas faire une soustraction de donnée train et validation)
 - *cabbi_train.p*: fichier pickle des id rp de train
 - *cabbi_val.p*: fichier pickle des id rp de validation
 - *dataset.csv*: données exportées depuis la base de données
 - *dataset_transformed.npy*: ficheir contenant les données préparées pour le modèle
 - *sirets_test.p*: fichier pickle des id sirus de test (peut être vide, dans ce cas faire une soustraction de donnée train et validation)
 - *sirets_train.p*: fichier pickle des id sirus de train
 - *sirets_val.p*:fichier pickle des id sirus de validation 
 - *output* et *tmp*: dossier contenant des données de training
 - *dict_info.pickle*: fichier pickle contenant les metaparams du modèle
 - *tokenizer.p*: fichier tokenize, contient le vocabulaire et l'encodage du dataset
 - *model_i*: dossier qui contient un modèle tensorflow entraîné
 - *df_solution_eval.p*: fichier généré par script_generate_groundtruth_file.py 
 - *df_solution_test.p*: fichier généré par script_generate_groundtruth_file.py 
 - *best_model_optuna.json*: paramêtres du metamodèle
 - *meta_model.p*: modele de codage auto

Doivent être créés à la main:
 - config_si.yaml: config de runtime si
 - config_bi.yaml: config de runtime bi
Ce sont des copies de la config de training dont on a enlevé les éléments et champs non applicables à l'un ou l'autre des types de documents.
Voir les exemples


## Training
----------

 1. Jeu de donnée sirus et BI chargé dans un postgresql
 2. Choix des différents cleaner, process, model dans *config.yaml*
 3. Execution de *script_pipeline.py*
 4. Etapes 1 - 3 à répeter jusqu'a satisfaction du modèle
 5. Création des fichiers config_si.yaml et config_bi.yaml
 6. Projection des données sirus dans postgresql (*script_project_dbtable.py*)
 7. Chargement des données dans ElasticSearch:
    - *script_export_to_es_bi_enriched_data.py*
    - *script_export_to_es_sirus_enriched_data.py*
    - *script_export_to_es_projections.py*
 8. Création des fichiers de ground_truth via *script_generate_groundtruth_file.py*
 9. Execution de *script_runtime.py* pour remplir une table SQL de résultats pré-métamodèle
 10. Prédictions NAF pour SIRUS dans la DB: *nomenclatures/script_project_table.py*
 11. Execution de *script_metamodel_optimisation.py* pour optimiser une nouveau méta modèle et créer une modèle de codage automatique
 12. Re-execution de *script_runtime.py* pour remplir une table SQL de résultats avec métamodèle et codage automatique
 13. Execution de *script_generate_scoring.py* pour générer des données d'analyse de résultats et quelques graphs 
 14. Analyse des résultats via *Exploration_Resultats.ipynb*
 
 
## Amélioration possible
------------------------

Dans le modèle de similarité, les matrices d'entrainement sont construites via le lot_id/vague_id.
Afin de maximiser la performance du modèle, la matrice doit contenir des éléments le plus proche possible.
L'idée est de changer la construction par lot_id par similarité ElasticSeatch (via une requête à définir)


Author : bsanchez@starclay.fr
date : 06/08/2020