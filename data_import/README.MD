PACKAGE DATA_IMPORT
===================


Ce package contient les scripts d'import de données.
Pour le moment, il gère l'import de données CSV vers une base postgreSQL.


Fichiers 
--------

- *config_import.yaml* : fichier contenant tout les csv à importer avec leur meta data
- *csv_to_postgresql.py* : script d'import de données depuis des fichiers CSV vers une base SQL.
    La source de donnée doit avoir un fichier descriptif de typage sur le modèle de typage_rp.csv  
    Ce script peut prendre un fichier yaml de configuration à la place de celui de base en appelant l'argument -c ou --config puis le nom du fichier souhaité.
    Un fichier de log appelé logCSVtoPosgresql.log est créé à l'endroit d'appel du script
- *typage_RP.csv*: fichier de typage pour l'import de données de type Individu
- *typage_sirus.csv*: fichier de typage pour l'import de données de type Etablissement sirus
- *typage_nomenclatures.csv*: fichier de typage pour l'import de données de type Nomenclature (liste des nomenclatures INSEE)
- *typage_modalites.csv*: fichier de typage pour l'import de données de type Modalite (valeurs des noeuds des nomenclatures)


Comment ajouter le support de nouveaux types de données
-------------------------------------------------------

Tout nouveau type de donnée doit avoir un fichier de typage.
Celui-ci contient une ligne de titre, puis plusieurs lignes de typage.
Une ligne de typage contient :
    - le nom du champs
    - le data type pandas du champ
    - le data type SQL du champ
    - une instruction SQL eventuelle (ex: "" si rien, "PRIMARY KEY", etc)
    



Author : cpoulet@starclay.fr
date : 16/07/2020