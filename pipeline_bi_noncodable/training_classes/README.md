PACKAGE PIPELINE
===================


Ce package contient les classes et script pour opérer des modèles de machine learning.
Gère le nettoyage et l'entrainement des données

IMPORTANT: suite à l'évolution du code pour se concentrer sur les modèles Deep Learning,
une partie des process ne sont pas mis à jour (le training fonctionne, la sauvegarde aussi, mais pas le chargement)


Fichiers 
--------
- *CleanerRemoveSpecialChar.py*: classe de nettoyage des inputs textuels

- *Model.py* : fichier contenant la classe mère des modèles
- *ModelAuto.py* : Modèle basé sur AutoSklearn
- *ModelLogistiqueRegression.py* : Modèle basé sur le GLM logistique régression sklearn
- *ModelLogistiqueRegressionCV.py* : Modèle basé sur le GLM logistique régression CV sklearn
- *ModelMLP.py* : Modèle Deep Learning à base de LSTM bidirectionnelles. Chaque champ est un input du modèle
- *ModelMLPFastText.py* : Modèle Deep Learning basé sur des LSTM bidirectionnelles et des embeddings FastText. Le modèle a un seul input qui est une concaténation des champs du BI
- *ModelMLPFusion.py* : Modèle Deep Learning basé sur des LSTM bidirectionnelles. Le modèle a un seul input qui est une concaténation des champs du BI
- *ModelNaiveBayes.py* : Modèle basé sur sklearn.naive_bayes.MultinomialNB
- *ModelSmote.py* : Modèle basé sur du sur-echantillonage SMOTE
- *ModelSVM.py* : Modèle basé sur sklearn.svm.LinearSVC (avec ou sans optimisation via hyperopt)
- *ModelTransformer.py* : Modèle Deep Learning basé sur les modèles Transformer
- *ModelTree.py* : Modèle basé sur sklearn.tree.DecisionTreeClassifier
- *ModelXGB.py* : Modèle basé sur XGBoost

- *preprocessing.py*: fichier de fonctions de nettoyage NLP

- *Process.py* : fichier contenant la classe mère des Process (transformation des données textuelles en données ingérables par un modèle)
- *ProcessBigram.py* : Transformation en bigrammes
- *ProcessDoc2Vec.py* : Transformation en Doc2Vec
- *ProcessEmbedding.py* : Transformation en vecteurs via entrainement d'une couche d'embeddings
- *ProcessFasttext.py* : Transformation en vecteurs via embeddings pré-entrainés fasttext
- *ProcessMLP.py* : Tokenisation pour modèle contenant la couche d'embedding
- *ProcessTfIdf.py* : Transformation TF-IDF

- *utils.py*: fichier de fonction utilitaire par rapport à la gestion des fichiers en local ou sur minio


Author : bsanchez@starclay.fr
date : 06/08/2020