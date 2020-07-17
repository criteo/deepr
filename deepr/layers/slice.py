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

    def __init__(self, **kwargs):
        super().__init__(n_in=2, n_out=1, **kwargs)

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer"""
        vectors, mask = tensors
        batch_size, sequence_length = tf.shape(vectors)[0], tf.shape(vectors)[1]
        indices = tf.tile(tf.expand_dims(tf.range(sequence_length), 0), [batch_size, 1])
        indices *= tf.cast(mask, tf.int32)
        lengths = tf.reduce_max(indices, axis=-1)
        positions = tf.stack([tf.range(0, batch_size), lengths], axis=1)
        return tf.gather_nd(vectors, tf.cast(positions, tf.int64))
