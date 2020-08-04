"""Triplet Precision Layer."""

import tensorflow as tf

from deepr.layers import base
from deepr.layers.reduce import WeightedAverage
from deepr.utils.broadcasting import make_same_shape


class TripletPrecision(base.Layer):
    """Triplet Precision Layer."""

    def __init__(self, **kwargs):
        super().__init__(n_in=4, n_out=1, **kwargs)

    def forward(self, tensors, mode: str = None):
        """Computes Triplet Precision

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
            BPR loss
        """
        # Retrieve positives and negatives logits
        positives, negatives, mask, weights = tensors
        positives, negatives = make_same_shape([positives, negatives], broadcast=False)

        # One triplet precision per event
        event_triplet = WeightedAverage()((tf.cast(positives > negatives, tf.float32), tf.cast(mask, tf.float32)), mode)

        # Each event contributes according to its weight
        event_weights = weights * tf.to_float(tf.reduce_any(mask, axis=-1))
        return tf.div_no_nan(tf.reduce_sum(event_triplet * event_weights), tf.reduce_sum(event_weights))
