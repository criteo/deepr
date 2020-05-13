"""Sparse Layers"""

from typing import Any

import tensorflow as tf

from deepr.layers import base


class ToDense(base.Layer):
    """Sparse to Dense Layer"""

    def __init__(self, default_value: Any, **kwargs):
        super().__init__(n_in=1, n_out=1, **kwargs)
        self.default_value = default_value

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer"""
        return tf.sparse.to_dense(tensors, default_value=self.default_value)
