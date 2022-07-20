from tensorflow import keras
import tensorflow as tf

from .nomenclature_distance import NomenclatureDistance


class SimilarityModel(keras.Model):
    def __init__(self, *args, nomenclature_distance: NomenclatureDistance, **kwargs):
        """
        Extension de keras.Model implémentant le training metric-learning en réseau siamois

        :param args: liste d'arguments passés à super()
        :param nomenclature_distance: NomenclatureDistance utilisée pour la distance
        :param kwargs: dictionnaire d'arguments passés à super()
        """
        super().__init__(*args, **kwargs)
        self.nomenclature_distance = nomenclature_distance
        
    def train_step(self, data):
        """ 
        training step
        """
        anchors, anchors_fields_embeddings, anchors_positions_embeddings = data[0], data[1], data[2]
        positives, positives_fields_embeddings, positives_positions_embeddings = data[3], data[4], data[5]
        class_ids = tf.cast(data[6][:, 0], dtype=tf.int32)

        with tf.GradientTape() as tape:
            # Run both anchors and positives through model.
            anchor_projection = self([anchors, anchors_fields_embeddings, anchors_positions_embeddings], 
                                     training=True)
            positive_projection = self([positives, positives_fields_embeddings, positives_positions_embeddings], 
                                       training=True)

            # Calculate cosine similarity between anchors and positives. As they have
            # been normalised this is just the pair wise dot products.
            similarities = self.similarity_func(anchor_projection, positive_projection)
            
            # Since we intend to use these as logits we scale them by a temperature.
            # This value would normally be chosen as a hyper parameter.
            # temperature = 0.2
            # similarities /= temperature

            # We use these similarities as logits for a softmax. The labels for
            # this call are just the sequence [0, 1, 2, ..., num_classes] since we
            # want the main diagonal values, which correspond to the anchor/positive
            # pairs, to be high. This loss will move embeddings for the
            # anchor/positive pairs together and move all other pairsapart.
            label_distances = self.nomenclature_distance.get_distance_mat_from_tf_indices(class_ids)
            loss = self.compiled_loss(label_distances, similarities, regularization_losses=self.losses)

        # Calculate gradients and apply via optimizer.
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        # Update and return metrics (specifically the one for the loss value).
        self.compiled_metrics.update_state(label_distances, similarities)
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        """
        Très similaires au train_step, cette fonction permet de produire les metriques de validations à partir des données de validation
        """
        # Unpack the data
        anchors, anchors_fields_embeddings, anchors_positions_embeddings = data[0], data[1], data[2]
        positives, positives_fields_embeddings, positives_positions_embeddings = data[3], data[4], data[5]
        class_ids = tf.cast(data[6][:, 0], dtype=tf.int32)

        anchor_projection = self([anchors, anchors_fields_embeddings, anchors_positions_embeddings], 
                                     training=False)
        positive_projection = self([positives, positives_fields_embeddings, positives_positions_embeddings], 
                                       training=False)
        
        similarities = self.similarity_func(anchor_projection, positive_projection)

            # Since we intend to use these as logits we scale them by a temperature.
            # This value would normally be chosen as a hyper parameter.
        # temperature = 0.2
        # similarities /= temperature
        
        label_distances = self.nomenclature_distance.get_distance_mat_from_tf_indices(class_ids)
        # Updates the metrics tracking the loss
        self.compiled_loss(label_distances, similarities, regularization_losses=self.losses)
        # Update the metrics.
        self.compiled_metrics.update_state(label_distances, similarities)
        # Return a dict mapping metric names to current value.
        # Note that it will include the loss (tracked in self.metrics).
        return {m.name: m.result() for m in self.metrics}
    
    @staticmethod
    def similarity_func(x_projections, y_projections):
        """
        Calcul de similarité entre 2 vecteurs

        :param x_projections: vecteur 1 (tensor)
        :param x_projections: vecteur 1 (tensor)
        :returns: distance (tensor)
        """
        return tf.einsum("ae,pe->ap", x_projections, y_projections)