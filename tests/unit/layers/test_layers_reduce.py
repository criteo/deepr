"""Tests for layers.reduce"""

import numpy as np
import tensorflow as tf

import deepr


def test_layers_weighted_average():
    """Test for WeightedAverage"""
    layer = deepr.layers.WeightedAverage()
    tensors = (tf.constant([[1, 2], [2, 1]], dtype=tf.float32), tf.constant([2, 3], dtype=tf.float32))
    result = layer(tensors)
    with tf.Session() as sess:
        got = sess.run(result)
        np.testing.assert_allclose(got, [8 / 5, 7 / 5])
