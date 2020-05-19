"""Slicing Layers"""

import tensorflow as tf

from deepr.layers import base


class Slice(base.Layer):
    """Slice Layer"""

    def __init__(self, begin: int, end: int, **kwargs):
        super().__init__(n_in=1, n_out=1, **kwargs)
        self.begin = begin
        self.end = end

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer"""
        return tensors[self.begin : self.end]


class SliceFirst(base.Layer):
    """Slice First Layer"""

    def __init__(self, size: int, **kwargs):
        super().__init__(n_in=1, n_out=1, **kwargs)
        self.size = size

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer"""
        return tensors[: self.size]


class SliceLast(base.Layer):
    """Slice First Layer"""

    def __init__(self, size: int, **kwargs):
        super().__init__(n_in=1, n_out=1, **kwargs)
        self.size = size

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer"""
        return tensors[-self.size :]


class SliceLastPadded(base.Layer):
    """Get the values that corresponds to the last not padded values"""

    def __init__(self, padded_value: int = -1, **kwargs):
        super().__init__(n_in=2, n_out=1, **kwargs)
        self.padded_value = padded_value

    def get_length(self, batch):
        positive_values = tf.cast(tf.math.not_equal(batch, self.padded_value), tf.int32)
        real_length = tf.reduce_sum(positive_values, -1)
        real_length = tf.cast(real_length, tf.int32)
        return real_length

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer"""
        vectors, product_indices = tensors

        # 1. Get indices of the last non '-1' value in each timeline
        lengths = self.get_length(product_indices)
        indices = lengths - tf.constant(1, dtype=lengths.dtype)

        # 2. Get coordinates of the last non '-1' value in each timeline (zip `indices` with range(batch_size))
        batch_size = tf.shape(vectors, out_type=tf.int64)[0]
        rang = tf.range(0, batch_size, dtype=tf.int64)

        indices = tf.cast(indices, rang.dtype)
        positions = tf.stack([rang, indices], axis=1)

        return tf.gather_nd(vectors, positions)
