"""String Layers"""

import tensorflow as tf

from deepr.layers import base
from deepr.utils.broadcasting import make_same_shape


class StringJoin(base.Layer):
    """String Join Layer"""

    def __init__(self, n_in: int = 2, separator: str = " ", **kwargs):
        super().__init__(n_in=n_in, n_out=1, **kwargs)
        self.separator = separator

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer"""
        tensors = [tensor if tensor.dtype == tf.string else tf.as_string(tensor) for tensor in tensors]
        tensors = make_same_shape(tensors)
        return tf.strings.join(tensors, separator=self.separator)
