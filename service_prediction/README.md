SERVICE DE PREDICTION - Build et déploiement
============================================

Le service de prédiction est construit avec fastapi, et lancé via uvicorn.


Fichiers
--------

- *src*: code du service
- *test*: tests du service
- *Dockerfile*: instruction de création de l'image docker
- *build_docker_image.sh*: script de création de l'image docker
- *requirements.txt*: fichier requirements pour pip dans l'image docker

- *api_functions.py*: fichier de fonctions d'appels de l'API
- *script_autocode_bis.py*: script permettant d'autocoder une table de BI en utilisant le service de prédiction.



Build
-----

Le service réutilise le code des diverses pipelines, aui doit pour cela être rendu accessible au contexte docker lors de l'appel de docker build.

Pour cela, appelez *build_docker_image.sh*: il copie les fichiers nécessaires, corrige les paths d'import dans les fichiers python, gère le tagging de l'image générée et le push eventuel vers un registry.

variables surchargeables:

- DOCKER_IMAGE_NAME: nom de l'image (défaut : ssplab/aiee2-prediction/service_prediction)
- DOCKER_IMAGE_TAG: tag de l'image (défaut: latest)
- PYTHON_PIP_PROXY: proxy à utiliser pour pip lors du build
- DOCKER_REGISTRY: à setter s'il faut pusher sur un registry, avec DOCKER_LOGIN et DOCKER_PWD eventuellement


Run
---

Le service doit pouvoir se connecter à la base Postgres et à minio pour démarrer, et à Elasticsearch pour les requètes SIRET.


Hors docker:

```
    cd src
    export AWS_ACCESS_KEY_ID=...
    export AWS_SECRET_ACCESS_KEY=...
    export AWS_SESSION_TOKEN=...
    uvicorn main_app:app
```

Avec docker:

```
    docker run -d \
        -e AWS_ACCESS_KEY_ID=... \
        -e AWS_SECRET_ACCESS_KEY=... \
        -e AWS_SESSION_TOKEN=...\
        ssplab/aiee2-prediction/service_prediction:latest
```

Autres variables d'environnement possibles (voir src/README.md pour liste complète)

- http_proxy=... si le service est derrière un proxy (c'est le cas sur la plateforme insee)
- APP_USERNAME/APP_PASSWORD pour surcharger / ajouter un compte utilisateur
- APP_MINIO_ENDPOINT pour configurer l'adresse minio
- APP_ELASTICSEARCH_HOSTSLIST pour configurer les adresses ES
- APP_BDD_MINIO_PATH_TO_SETTINGS pour configurer l'adresse du fichier settings sur minio contenant les données de connexion à Postgresql


Deploiement INSEE
-----------------

Le déploiement dans le cadre de l'Insee se fait en 2 parties:

- build de l'image et push vers un registry via la CI gitlab (voir ../.gitlab_ci.yaml)
- déploiement dans le cluster via marathon (contrat à venir)


Appels
------

Pour consulter le service, la 1e étape est d'obtenir un token. Cela se fait via /token - POST, en passant dans la requête data={'username': login, "password": mdp}
Si le username est connu et que le password matche, la réponse sera un json {'token_type': 'bearer', 'access_token': str}

Pour tous les appels prédictifs, il faut alors ajouter un header {'Authorization': f'{token_type} {access_token}'}

Tous les appels prédictifs prennent en entrée le même format:
    json={"documents": json_documents} avec json_documents une liste de json contenant les champs:

    - cabbi: str = Field(None, title="cabbi du BI", max_length=20)
    - rs_x: str = Field(None, title="nom déclaratif de l'entreprise", max_length=300)
    - actet_x: str = Field(None, title="activité déclarative de l'entreprise", max_length=300)
    - profs_x: Optional[str] = Field(None, title="profession salarié déclarée", max_length=300)
    - profi_x: Optional[str] = Field(None, title="profession indépendante déclarée", max_length=300)
    - profa_x: Optional[str] = Field(None, title="ancienne profession déclarée", max_length=300)
    - numvoi_x: str = Field('', title="Numéro de l'adresse déclarée du lieu de travail", max_length=15)
    - bister_x: Optional[str] = Field('', title="Indice de repetition", max_length=30)
    - typevoi_x: Optional[str] = Field('', title="Type de voie de l'adresse déclarée du lieu de travail", - max_length=100)
    - nomvoi_x: Optional[str] = Field('', title="Nom de voie de l'adresse déclarée du lieu de travail", - max_length=100)
    - cpladr_x: Optional[str] = Field('', title="Complement d'adresse'", max_length=100)
    - clt_x: Optional[str] = Field('', title="commune déclarée du lieu de travail", max_length=100)
    - dlt_x: Optional[str] = Field('', title="département du lieu de travail", max_length=100)
    - plt_x: Optional[str] = Field('', title="Pays du lieu de travail", max_length=100)
    - depcom_code: str = Field(None, title="Code departement-commune", max_length=10)
    - vardompart_x: Optional[str] = Field('', title="Travaille a un endroit variable ou a domicile", - max_length=10)

(voir src/app_prediction#BIModelInput)

Tous les points supportent des appels contenant plusieurs documents, mais pour des raisons de mémoire dans les modèles, évitez d'être trop gourmands (50 par 50 -> ok, 500 par 500 -> probable problème de mémoir")


Les points api de prédictions sont:

- /noncodable - GET:
    Renvoie {'noncodable': [dict]} avec 
        dict = {'cabbi': cabbi, 'status': 'ok', 'non_codable': bool, 'predictions': [score_0, score_1]}

- /naf - GET:
    Renvoie {'naf': [dict]} avec 
        dict = {'cabbi': cabbi, 'status': 'ok', 'predictions': [[code_0, score_0], ...]} ou
        dict = {'cabbi': cabbi, 'status': 'error', 'error': 'message d'erreur'}

- /pcs - GET:
    Renvoie {'pcs': [dict]} avec 
        dict = {'cabbi': cabbi, 'status': 'ok', 'predictions': [[code_0, score_0], ...]} ou
        dict = {'cabbi': cabbi, 'status': 'error', 'error': 'message d'erreur'}

- /siret - GET:
    Renvoie {'siret': [dict]} avec 
        dict = {'cabbi': cabbi, 'status': 'ok', 'predictions': [[code_0, score_0], ...], 'codage_auto': bool, 'codage_auto_proba':float} ou
        dict = {'cabbi': cabbi, 'status': 'error', 'error': 'message d'erreur'}

Pour plus de détail, se référer au swagger fastapi (lancer le service, puis ouvrir adresse_du_service/doc)