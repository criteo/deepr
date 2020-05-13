"""BPR Loss Layer"""

import tensorflow as tf

from deepr.layers import base
from deepr.layers.reduce import WeightedAverage, Average
from deepr.utils.broadcasting import make_same_shape


class BPR(base.Layer):
    """Vanilla BPR Loss Layer"""

    def __init__(self, **kwargs):
        super().__init__(n_in=2, n_out=1, **kwargs)

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer

        Parameters
        ----------
        tensors : Tuple[tf.Tensor]
            - positives: shape = [batch]
            - negatives: shape = [batch]

        Returns
        -------
        tf.Tensor
            BPR loss
        """
        positives, negatives = tensors
        positives, negatives = make_same_shape([positives, negatives], broadcast=False)
        losses = -tf.log_sigmoid(positives - negatives)
        return Average()(losses, mode)


class MaskedBPR(base.Layer):
    """Masked BPR Loss Layer"""

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
            BPR loss
        """
        positives, negatives, mask, weights = tensors
        positives, negatives = make_same_shape([positives, negatives], broadcast=False)
        # One score per negative : (batch, num_events, num_negatives)
        scores = -tf.log_sigmoid(positives - negatives)
        # One loss per event, average of scores : (batch, num_events)
        event_scores = WeightedAverage()((scores, tf.to_float(mask)))
        # Each event contributes according to its weight
        event_weights = weights * tf.to_float(tf.reduce_any(mask, axis=-1))
        event_losses = event_scores * event_weights
        return tf.div_no_nan(tf.reduce_sum(event_losses), tf.reduce_sum(event_weights))
