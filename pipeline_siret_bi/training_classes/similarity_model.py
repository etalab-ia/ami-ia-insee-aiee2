import numpy as np
import tensorflow as tf
from tensorflow import keras


class SimilarityModel(keras.Model):

    def __init__(self,  *args, **kwargs):
        """
        Classe keras qui gère le training en mode siamois (distance learning)
        """
        super().__init__(*args, **kwargs)

    def test_step(self, data):
        """
        Très similaires au train_step, cette fonction permet de produire les metriques de validations à partir des données de validation
        
        :param data: AnchorPositivePairs
        """
        # Unpack the data
        anchors, anchors_fields_embeddings, anchors_positions_embeddings = data[0], data[1], data[2]
        positives, positives_fields_embeddings, positives_positions_embeddings = data[3], data[4], data[5]
        sirets = tf.cast(data[6][:, 0], dtype=tf.int64)
        
        anchor_projection = self([anchors, anchors_fields_embeddings, anchors_positions_embeddings], 
                                     training=False)
        positive_projection = self([positives, positives_fields_embeddings, positives_positions_embeddings], 
                                       training=False)
        batch_size = tf.shape(anchors)[0]
        lab = sirets.numpy() 
        matrix = np.identity(batch_size, dtype = int)
        for x in range(batch_size):
            for y in range(batch_size):
                if lab[x] == lab[y]:
                    matrix[x,y] = 1
        label_distances = tf.convert_to_tensor(matrix, dtype=None, dtype_hint=None, name=None)
        
        similarities = tf.einsum(
                "ae,pe->ap", anchor_projection, positive_projection
            )

        y_pred = similarities
        self.compiled_loss(label_distances, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(label_distances, y_pred)

        return {m.name: m.result() for m in self.metrics}
        
    def train_step(self, data):
        """
        Organise l'apprentisssage du modèle keras. 
        On possède une paire de donnée : une donnée exemple et son match "postive anchor".
        On indique aux modèle d'unpack l'exemple et son anchor.
        Puis, on les faits traverser le modèle.
        Puis on construit une matrice de simalirité.
        On calcule le coût de cette matrice de similarité avec la matrice cible.
        
        Matrice cible : 
        SIRUS
            B  [1 0 0]
            I  [0 1 0]
               [0 0 1]  (Ressemble fortement à une matrice de similarité)
        
        On met 1 quand le pour les couple BI/Siret, 0 sinon.
        Cas particulier : Dans une même matrice, on peut avoir plusieurs BI pour un même siret (même label)
        =>  On adapte les targets (sirus a deux fois le même label içi)
                SIRUS
            B  [1 1 0]
            I  [1 1 0]
               [0 0 1]
               
               
        On met à jour les métriques.

        :param data: AnchorPositivePairs
        """
        # Note: Workaround for open issue, to be removed.
        anchors, anchors_fields_embeddings, anchors_positions_embeddings = data[0], data[1], data[2]
        positives, positives_fields_embeddings, positives_positions_embeddings = data[3], data[4], data[5]
        sirets = tf.cast(data[6][:, 0], dtype=tf.int32) 
        
        batch_size = tf.shape(anchors)[0]
        matrix = np.identity(batch_size, dtype = int)
        lab = sirets.numpy() 
        for x in range(batch_size):
            for y in range(batch_size):
                if lab[x] == lab[y]:
                    matrix[x,y] = 1
        label_distances = tf.convert_to_tensor(matrix, dtype=None, dtype_hint=None, name=None)
        
        with tf.GradientTape() as tape:
            # Run both anchors and positives through model.
            anchor_projection = self([anchors, anchors_fields_embeddings, anchors_positions_embeddings], 
                                     training=True)
            positive_projection = self([positives, positives_fields_embeddings, positives_positions_embeddings], 
                                       training=True)
            similarities = tf.einsum(
                "ae,pe->ap", anchor_projection, positive_projection
            )
            loss = self.compiled_loss(label_distances, similarities)

        # Calculate gradients and apply via optimizer.
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update and return metrics (specifically the one for the loss value).
        self.compiled_metrics.update_state(label_distances, similarities)
        return {m.name: m.result() for m in self.metrics}
    
