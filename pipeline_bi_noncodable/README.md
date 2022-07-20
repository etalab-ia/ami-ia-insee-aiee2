PACKAGE PIPELINE
===================


Ce package contient les classes et script pour opéré des modèles de machine learning.
Gère le nettoyage et l'entrainement des données


Fichiers 
--------
- *config.yaml* : fichier contenant toutes les opération DataEng/ML à lancer
- *launch* : fichier pour lancer plusieurs fois le pipeline
- *training_classes* : dossier contenant les classes Modèles et manipulation/transformation de données
- *script_pipeline.py* : script qui gère l'orchestration de l'execution d'un training à partir d'un fichier de config
- *script_runtime.py* : script qui gère la partie runtime : chargement des éléments de la pipeline, et prédiction à partir d'un input "naturel" (un BI non transformé)
- *script_publish_model.py* : fichier qui permet de publier un modèle sur minio à vocation d'être intégré dans le service de prédiction
- *choose_threshold.ipynb* : notebook permettant de charger le modèle entraîné, de visualiser l'impact des valeurs de threshold sur la précision et le recall, de choisir le threshold correct et de le sauvegarder dans la config du modèle


Fonctionnement général
----------------------

Un fichier de configuration contient les opérations et leurs paramètres à éxécuter. Une opération est de type:
    - cleaner: nettoyage de données
    - Process: Opération qui va modifier les données, peut avoir une modèle s'apparente à la phase de feature engineering
    - Modèle: Modèle de machine learning et métriques qui est automatique sauvegarder

Le fichier pipeline se charge de l'orchestration de ces opérations.

La sauvegarde sur minio est réalisé ainsi : Chaque run de pipeline crée un dossier unique correspond au jour du début du run et du nombre de run ayant été réalisé ce jour-là. Tout les fichiers du run seront estampillé de cet id, dans ce dossier. L'ID est donc unique.

Le pipeline peut être executé de 3 façon différente: 
    -  Uniquement avec les opération de cleaner et de transformer, les fichiers sont sauvegarder dans un dossier commun ("based") et sont destiné à être réutiliser. Attention tout les fichiers déjà présent dans le dossier based seront supprimé.  
    - Uniquement avec les opération d'échantillonage et modèle. Dans ce cas les données sont chargé depuis le fichier commun ("based") et les modèles sauvegardé dans leur dossier unique (paragraphe de dessus).
    - 2en1 (par défaut), le pipeline est éxécuter du début jusqu'a la fin. (rien n'est sauvegarder dans based)

Les modèles qui peuvent être tuné le sont via hyperopt (optimisation bayesien).


Une fois la pipeline entraînée, elle peut être testée via script_runtime.py sur des données non transformées (ces fonctions sont appelées dans le service de prédiction).


Enfin, le modèle entraîné peut être publié via script_publish_model.py pour permettre son déploiement dans le service de prédiction.


IMPORTANT : Du à l'évolution du code vers des modèles de type Transformer, les process autres que ProcessMLP peuvent être correctement entraînés via script_pipeline.py, mais les fonctions de script_runtime.py ne fonctionnement pas avec ces classes (ce qui par ricochet empèche le fonctionnement de tout autre modèle que les ModelMLP... et ModelTransformer). Elles ne sont donc pas non plus déployables en l'état dans le service de prédiction.


IMPORTANT : Pour utiliser fasttext, ProcessMLP devrait normalement télécharger si besoin le modèle fr, puis le resizer à la bonne taille. Si pour une raison ou une autre ce téléchargement ne fonctionne pas, le fichier cc.fr.{embeddings_size}.bin doit se trouver dans le dossier fasttext.
Pour que le resizing fonctionne, il doit être fait sur un modèle 300 qu'on vient de télécharger (le passage via minio modifie quelque chose, ce qui fait par la suite échouer le resize)

Author : bsanchez@starclay.fr
date : 06/08/2020