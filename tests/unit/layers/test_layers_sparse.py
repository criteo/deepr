"""Tests for layers.sparse"""

import numpy as np
import tensorflow as tf

import deepr


def test_layers_sparse_to_dense():
    """Test for ToDense"""
    layer = deepr.layers.ToDense(default_value=-1)
    tensors = tf.sparse.SparseTensor(indices=[[0, 1], [1, 0]], values=[1, 1], dense_shape=[2, 2])
    result = layer(tensors)
    with tf.Session() as sess:
        got = sess.run(result)
        expected = [[-1, 1], [1, -1]]
        np.testing.assert_equal(got, expected)
