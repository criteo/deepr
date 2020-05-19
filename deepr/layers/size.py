"""Size Layers"""

import tensorflow as tf

from deepr.layers import base


class IsMinSize(base.Layer):
    """Compare size of inputs to minimum"""

    def __init__(self, size: int, **kwargs):
        super().__init__(n_in=1, n_out=1, **kwargs)
        self.size = size

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer"""
        return tf.greater_equal(tf.size(tensors), self.size)
