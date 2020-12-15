"""TopOne Max Loss Layer"""

import tensorflow as tf

from deepr.layers import base
from deepr.layers.reduce import WeightedAverage, Average
from deepr.layers.core import Softmax
from deepr.utils.broadcasting import make_same_shape


class TopOneMax(base.Layer):
    """Vanilla TopOne Max Loss Layer"""

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
            TopOne Max loss
        """
        positives, negatives = tensors
        positives, negatives = make_same_shape([positives, negatives], broadcast=False)
        softmax_scores = Softmax()((negatives, tf.ones_like(negatives)))
        losses = tf.multiply(softmax_scores, tf.nn.sigmoid(negatives - positives) + tf.nn.sigmoid(tf.square(negatives)))
        return Average()(losses, mode)


class MaskedTopOneMax(base.Layer):
    """Masked TopOne Max Loss Layer"""

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
            TopOne Max loss
        """
        positives, negatives, mask, weights = tensors
        positives, negatives = make_same_shape([positives, negatives], broadcast=False)
        softmax_scores = Softmax()((negatives, tf.cast(mask, dtype=tf.float32)))
        losses = tf.multiply(softmax_scores, tf.nn.sigmoid(negatives - positives) + tf.nn.sigmoid(tf.square(negatives)))
        # One loss per event, average of scores : (batch, num_events)
        event_scores = WeightedAverage()((losses, tf.cast(mask, dtype=tf.float32)))
        # Each event contributes according to its weight
        event_weights = weights * tf.cast(tf.reduce_any(input_tensor=mask, axis=-1), dtype=tf.float32)
        event_losses = event_scores * event_weights
        return tf.math.divide_no_nan(tf.reduce_sum(input_tensor=event_losses), tf.reduce_sum(input_tensor=event_weights))
