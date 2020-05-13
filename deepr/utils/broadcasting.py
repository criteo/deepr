"""Tensorflow Broadcasting utilities"""

from typing import List

import tensorflow as tf


def make_same_shape(tensors: List[tf.Tensor], broadcast: bool = True) -> List[tf.Tensor]:
    """Make list of tensors the same shape

    Parameters
    ----------
    tensors : List[tf.Tensor]
        List of tensors
    broadcast : bool, optional
        If True, not only add missing dims, also broadcast

    Returns
    -------
    List[tf.Tensor]
    """
    biggest = sorted(tensors, key=lambda t: len(t.shape))[-1]

    def _add_missing_dims(t: tf.Tensor, ndims: int):
        if ndims == 0:
            return t
        return _add_missing_dims(tf.expand_dims(t, -1), ndims - 1)

    tensors = [_add_missing_dims(t, len(biggest.shape) - len(t.shape)) for t in tensors]
    if broadcast:
        tensors = [tf.broadcast_to(t, tf.shape(biggest)) for t in tensors]
    return tensors
