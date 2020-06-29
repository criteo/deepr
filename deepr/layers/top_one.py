"""Top1 Loss Layer"""

import tensorflow as tf

from deepr.layers import base
from deepr.layers.reduce import WeightedAverage, Average
from deepr.utils.broadcasting import make_same_shape


class TopOne(base.Layer):
    """Vanilla Top1 Loss Layer"""

    def __init__(self, bpr_max_regularizer=0.0, **kwargs):
        self.bpr_max_regularizer = bpr_max_regularizer
        super().__init__(n_in=2, n_out=1, **kwargs)

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer

        Parameters
        ----------
        tensors : Tuple[tf.Tensor]
            - positives : shape = (batch, num_events)
            - negatives : shape = (batch, num_events, num_negatives)

        Returns
        -------
        tf.Tensor
            Top1 loss
        """
        positives, negatives = tensors
        positives, negatives = make_same_shape([positives, negatives], broadcast=False)
        losses = tf.nn.sigmoid(negatives - positives) + tf.nn.sigmoid(tf.square(negatives))
        return Average()(losses, mode)


class MaskedTopOne(base.Layer):
    """Masked Top1 Loss Layer"""

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
            Top1 loss
        """
        positives, negatives, mask, weights = tensors
        positives, negatives = make_same_shape([positives, negatives], broadcast=False)
        losses = tf.nn.sigmoid(negatives - positives) + tf.nn.sigmoid(tf.square(negatives))
        event_scores = WeightedAverage()((losses, tf.to_float(mask)))
        # Each event contributes according to its weight
        event_weights = weights * tf.to_float(tf.reduce_any(mask, axis=-1))
        event_losses = event_scores * event_weights
        return tf.div_no_nan(tf.reduce_sum(event_losses), tf.reduce_sum(event_weights))
