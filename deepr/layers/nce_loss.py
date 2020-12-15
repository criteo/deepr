"""Negative Sampling Loss Layer"""

import tensorflow as tf

from deepr.layers import base
from deepr.layers.reduce import Average, WeightedAverage


class NegativeSampling(base.Layer):
    """Vanilla Negative Sampling Loss Layer.Loss

    Expected value at beginning of training : -2 * log(0.5) = 1.38
    """

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
        sampled_losses = tf.reduce_sum(input_tensor=sampled_losses, axis=2)
        losses = tf.add(true_losses, sampled_losses)
        return Average()(losses, mode)


class MaskedNegativeSampling(base.Layer):
    """Masked Negative Sampling Loss Layer.Loss

    Expected value at beginning of training : -2 * log(0.5) = 1.38
    """

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
        event_scores = true_losses + WeightedAverage()((sampled_losses, tf.cast(mask, dtype=tf.float32)))
        event_weights = weights * tf.cast(tf.reduce_any(input_tensor=mask, axis=-1), dtype=tf.float32)
        return tf.math.divide_no_nan(tf.reduce_sum(input_tensor=event_scores * event_weights), tf.reduce_sum(input_tensor=event_weights))
