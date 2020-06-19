"""Losses Layer"""

import tensorflow as tf

from deepr.layers import base
from deepr.layers.reduce import WeightedAverage, Average
from deepr.utils.broadcasting import make_same_shape


# noinspection PyTypeChecker
class BPRMax(base.Layer):
    """Vanilla BPR Max Loss Layer"""

    def __init__(self, bpr_max_regularizer=0.0, **kwargs):
        self.bpr_max_regularizer = bpr_max_regularizer
        super().__init__(n_in=2, n_out=1, **kwargs)

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer

        Parameters
        ----------
        tensors : Tuple[tf.Tensor]
            - positives: shape = [batch, num_events, embedding_dim]
            - negatives: shape = [batch, num_events, num_negatives, embedding_dim]

        Returns
        -------
        tf.Tensor
            BPR Max loss
        """
        positives, negatives = tensors
        positives, negatives = make_same_shape([positives, negatives], broadcast=False)
        softmax_scores = self._compute_softmax_scores(negatives, tf.ones_like(negatives))
        losses = -tf.log(tf.reduce_sum(tf.multiply(softmax_scores, tf.nn.sigmoid(positives - negatives)), -1))
        # add bpr_max regularisation
        bpr_regularization = tf.multiply(
            tf.constant(self.bpr_max_regularizer, dtype=tf.float32),
            tf.reduce_sum(tf.multiply(softmax_scores, tf.square(negatives)), -1),
        )
        scores = losses + bpr_regularization
        return Average()(scores, mode)

    @staticmethod
    def _compute_softmax_scores(tensor: tf.Tensor, mask: tf.Tensor):
        """ Compute softmax"""
        rj_exp = tf.exp(tensor - tf.reduce_max(tensor, axis=2, keepdims=True))
        sum_exp_negative = tf.reduce_sum(tf.multiply(rj_exp, mask), axis=2, keep_dims=True)
        softmax_scores = tf.divide(rj_exp, (sum_exp_negative + 1e-8))
        return softmax_scores


class MaskedBPRMax(BPRMax):
    """Masked BPR Max Loss Layer"""

    def __init__(self, bpr_max_regularizer=0.0, **kwargs):
        self.bpr_max_regularizer = bpr_max_regularizer
        super().__init__(n_in=4, n_out=1, **kwargs)

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer
        (details: https://arxiv.org/pdf/1706.03847.pdf)

        Parameters
        ----------
        tensors : Tuple[tf.Tensor]
            - positives : shape = (batch, num_events)
            - negatives : shape = (batch, num_events, num_negatives)
            - mask : shape = (batch, num_events, num_negatives)
            - weights : shape = (batch, num_events)

        Returns
        -------
        tf.Tensor
            BPR Max loss
        """
        positives, negatives, mask, weights = tensors
        positives, negatives = make_same_shape([positives, negatives], broadcast=False)
        no_sampled_logits = tf.cast(tf.greater_equal(tf.reduce_sum(mask, -1), 0), tf.float32)
        softmax_scores = self._compute_softmax_scores(negatives, mask)
        # compute bpr_max losses
        losses = -tf.multiply(
            no_sampled_logits,
            tf.log(
                tf.reduce_sum(tf.multiply(tf.multiply(softmax_scores, tf.nn.sigmoid(positives - negatives)), mask), 2)
                + 1e-8
            ),
        )
        # compute regularization part
        bpr_regularization = tf.multiply(
            tf.constant(self.bpr_max_regularizer, dtype=tf.float32),
            tf.reduce_sum(tf.multiply(tf.multiply(softmax_scores, tf.square(negatives)), mask), 2),
        )
        losses_with_regularization = losses + bpr_regularization
        # One loss per event, average of scores : (batch, num_events)
        event_scores = WeightedAverage()((losses_with_regularization, tf.to_float(mask)))
        # Each event contributes according to its weight
        event_weights = weights * tf.to_float(tf.reduce_any(mask, axis=-1))
        event_losses = event_scores * event_weights
        return tf.div_no_nan(tf.reduce_sum(event_losses), tf.reduce_sum(event_weights))


class NegativeSampling(base.Layer):
    """Vanilla Negative Sampling Loss Layer"""

    def __init__(self, **kwargs):
        super().__init__(n_in=2, n_out=1, **kwargs)

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer

        Parameters
        ----------
        tensors : Tuple[tf.Tensor]
            - positives: shape = [batch, num_events, embedding_dim]
            - negatives: shape = [batch, num_events, num_negatives, embedding_dim]

        Returns
        -------
        tf.Tensor
            Negative Sampling loss
        """
        positives, negatives = tensors
        true_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(positives), logits=positives)
        sampled_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(negatives), logits=negatives)
        sampled_losses = tf.reduce_sum(sampled_losses, axis=2)
        losses = tf.add(true_losses, sampled_losses)
        return Average()(losses, mode)


class MaskedNegativeSampling(base.Layer):
    """Masked Negative Sampling Loss Layer"""

    def __init__(self, **kwargs):
        super().__init__(n_in=4, n_out=1, **kwargs)

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer

        Parameters
        ----------
        tensors : Tuple[tf.Tensor]
            - positives : shape = (batch, num_events, embedding_dim)
            - negatives : shape = (batch, num_events, num_negatives, embedding_dim)
            - mask : shape = (batch, num_events, num_negatives)
            - weights : shape = (batch, num_events)

        Returns
        -------
        tf.Tensor
            Negative Sampling  loss
        """
        positives, negatives, mask, weights = tensors
        true_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(positives), logits=positives)
        sampled_losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(negatives), logits=negatives)
        # filter the values that correspond to mask
        negative_number = tf.reduce_sum(mask, -1)
        sampled_losses = tf.multiply(sampled_losses, mask)
        sampled_losses = tf.reduce_sum(sampled_losses, axis=2)
        sampled_losses = sampled_losses / (negative_number + 1e-8)

        losses = tf.add(true_losses, sampled_losses)
        # One loss per event, average of scores : (batch, num_events)
        event_scores = WeightedAverage()((losses, tf.to_float(mask)))
        # Each event contributes according to its weight
        event_weights = weights * tf.to_float(tf.reduce_any(mask, axis=-1))
        event_losses = event_scores * event_weights
        return tf.div_no_nan(tf.reduce_sum(event_losses), tf.reduce_sum(event_weights))
