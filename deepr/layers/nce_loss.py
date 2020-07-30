"""Negative Sampling Loss Layer"""

import tensorflow as tf

from deepr.layers import base
from deepr.layers.reduce import Average


class NegativeSampling(base.Layer):
    """Vanilla Negative Sampling Loss Layer"""

    def __init__(self, **kwargs):
        super().__init__(n_in=2, n_out=1, **kwargs)

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer
        (details:
        https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)

        Parameters
        ----------
        tensors : Tuple[tf.Tensor]
            - positives : shape = (batch, num_events)
            - negatives : shape = (batch, num_events, num_negatives)

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
            - positives : shape = (batch, num_events)
            - negatives : shape = (batch, num_events, num_negatives)
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
        negative_number = tf.reduce_sum(tf.to_float(mask), -1)
        sampled_losses = tf.multiply(sampled_losses, tf.to_float(mask))
        sampled_losses = tf.reduce_sum(sampled_losses, axis=2)
        sampled_losses = tf.div_no_nan(sampled_losses, negative_number)

        losses = tf.add(true_losses, sampled_losses)
        # One loss per event, average of scores : (batch, num_events)
        # TODO: fix this line, it seems it's doing averaging twice
        # event_scores = WeightedAverage()((losses, tf.to_float(mask)))
        event_scores = losses
        # Each event contributes according to its weight
        event_weights = weights * tf.to_float(tf.reduce_any(mask, axis=-1))
        event_losses = event_scores * event_weights
        return tf.div_no_nan(tf.reduce_sum(event_losses), tf.reduce_sum(event_weights))
