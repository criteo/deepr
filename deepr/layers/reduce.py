"""Reduce Layers"""

import tensorflow as tf

from deepr.layers import base
from deepr.utils.broadcasting import make_same_shape


class Average(base.Layer):
    """Average Layer"""

    def __init__(self, **kwargs):
        super().__init__(n_in=1, n_out=1, **kwargs)

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer"""
        return tf.div_no_nan(tf.reduce_sum(tensors), tf.reduce_sum(tf.ones_like(tensors)))


class WeightedAverage(base.Layer):
    """Weighted Average Layer"""

    def __init__(self, default=0, **kwargs):
        super().__init__(n_in=2, n_out=1, **kwargs)
        self.default = default

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer"""
        values, weights = tensors
        axis = len(weights.shape) - 1
        values, weights = make_same_shape([values, weights], broadcast=False)
        weighted_values = tf.reduce_sum(values * weights, axis=axis)
        sum_weights = tf.reduce_sum(weights, axis=axis)
        if self.default is None:
            return weighted_values / sum_weights
        elif self.default == 0:
            weighted_average = tf.div_no_nan(weighted_values, sum_weights)
        else:
            weighted_average = tf.where(
                tf.equal(sum_weights, 0), self.default * tf.ones_like(weighted_values), weighted_values / sum_weights
            )
        return weighted_average
