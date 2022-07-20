SERVICE DE PREDICTION - Sources
===============================


fichiers
--------

- *backend_utils/config_parser.py*: utilitaires de parsing de config et de variables d'env
- *security*: gestion des utilisateurs et des tokens
- *config.yaml*: config de l'appli
- *config_env.yaml*: déclaration des variables d'environnement pour surcharger la config
- *logging.ini* configuration logging
- *main_app.py*: déclaration de l'application fastapi et des endpoints
- *app_prediction.py*: Fonctions relatives aux modèles: chargement et prédiction
- *passwords.json*: fichier contenant les utilisateurs connus (par défaut)

Les dossiers suivants sont copiés au build: *data_import*, *nomenclatures*, *pipeline_bi_noncodable*, *pipeline_siret_bi*.


Configuration
-------------

```
    app:
        log_level: INFO                                     // niveau de log
        security:
            password_file: passwords.json                   // fichier utilisateurs
            username: admin_recap                           // username connu au démarrage (s'ajoute ou surcharge ceux dans password_file, peut être nul)
            password:                                       // password du username connu au démarrage (s'ajoute ou surcharge ceux dans password_file, peut être nul)
            token_algorithm: HS256                          // algo token
            token_secret_key: ...
            token_lifetime_in_min: 30                       // lifetime du token en minutes

    models:
        local_directory: models                             // dossier local de téléchargement des modèles)             
        minio_endpoint: ...                                 // url minio (les credentials sont passés en variables d'env)
        naf_model: ...                                      // chemin minio vers dossier du modèle naf
        pcs_model: ...                                      // chemin minio vers dossier du modèle pcs
        siret_model: ...                                    // chemin minio vers dossier du modèle siret
        noncodables_model: ...                              // chemin minio vers dossier du modèle noncodable
    elasticsearch:
        host: str ou list[str]                              // url du ou des serveurs ES
        port: 9200                                          // port à utiliser
        index_bi: rp_2019_e                                 // index pour les requètes sur les BI
        index_sirus: sirus_2019_e                           // index pour les requètes sur Sirus

    bdd:
        minio_path_to_settings_file: ...                    // chemin minio vers le fichier contenant les settings de la BDD
        sirus_table: siret_2019                             // table sirus à utiliser
        sirus_proj_table: sirus_projection_2                // table de projections sirus à utiliser
        naf_proj_table: naf_projections_2019                // tables de projections NAF à utiliser

    addok_apis:                                             // urls des services de géocodage INSEE
        ban: http://api-ban.alpha.innovation.insee.eu/search
        bano: http://api-bano.alpha.innovation.insee.eu/search
        poi: http://api-poi.alpha.innovation.insee.eu/search
```

Voir config_env.yaml pour les variables d'environnement permettant de surcharger chaque élément de la config.