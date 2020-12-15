"""Rank Layer"""

import tensorflow as tf

from deepr.layers import base
from deepr.utils.broadcasting import make_same_shape


class ClickRank(base.Layer):
    """Click Rank Layer"""

    def __init__(self, **kwargs):
        super().__init__(n_in=3, n_out=1, **kwargs)

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer

        Parameters
        ----------
        tensors : Tuple[tf.Tensor]
            - positives: shape = (batch, num_events)
            - negatives: shape = (batch, num_events, num_negatives)
            - mask: shape = (batch, num_events, num_negatives)

        Returns
        -------
        tf.Tensor
            ClickRank
        """
        positives, negatives, mask = tensors
        # One score per negative : (batch, num_events, num_negative)
        positives, negatives = make_same_shape([positives, negatives], broadcast=False)
        positives_greater_negatives = tf.greater(positives, negatives)
        # One score per event, average of ranks : (batch, num_events)
        eps = 1e-8
        mask_float = tf.cast(mask, dtype=tf.float32)
        negatives_sum = tf.reduce_sum(input_tensor=tf.cast(positives_greater_negatives, dtype=tf.float32) * mask_float, axis=-1)
        # In case no negatives, click rank would be 0.5 (random).
        # Events with no negatives are then removed via masking, so it
        # should not impact the final loss in any way.
        event_ranks = 1.0 - (negatives_sum + eps) / (tf.reduce_sum(input_tensor=mask_float, axis=-1) + eps * 2)
        # Each event contributes according to it weight
        event_mask = tf.cast(tf.reduce_any(input_tensor=mask, axis=-1), dtype=tf.float32)
        event_ranks = event_ranks * event_mask
        return tf.reduce_sum(input_tensor=event_ranks) / tf.reduce_sum(input_tensor=event_mask)
