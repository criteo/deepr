"""Dropout Layers"""

import logging
import tensorflow as tf

from deepr.layers import base


LOGGER = logging.getLogger(__name__)


class SpatialDropout1D(base.Layer):
    """1D Dropout Layer"""

    def __init__(self, dropout_rate: float = 0.0, **kwargs):
        super().__init__(n_in=1, n_out=1, **kwargs)
        self.dropout_rate = float(dropout_rate)

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer"""
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        return tf.keras.layers.SpatialDropout1D(rate=self.dropout_rate)(tensors, training=is_training)


class Dropout(base.Layer):
    """ Dropout Layer"""

    def __init__(self, dropout_rate: float = 0.0, **kwargs):
        super().__init__(n_in=1, n_out=1, **kwargs)
        self.dropout_rate = float(dropout_rate)

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer"""
        is_training = mode == tf.estimator.ModeKeys.TRAIN
        return tf.keras.layers.Dropout(rate=self.dropout_rate)(tensors, training=is_training)
