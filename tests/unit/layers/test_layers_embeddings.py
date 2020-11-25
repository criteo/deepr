"""Tests for layers.embeddings"""

import numpy as np
import tensorflow as tf

import deepr


def test_layers_embeddings():
    """Test for Embedding"""
    layer = deepr.layers.Embedding(variable_name="embeddings", shape=(2, 10))
    layer2 = deepr.layers.Embedding(variable_name="embeddings", shape=(2, 10), reuse=True)
    result = layer(tf.constant([0, 1, -1]))
    result2 = layer(tf.constant([0, 1, -1]), reuse=True)
    result3 = layer2(tf.constant([0, 1, -1]))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        got = sess.run(result)
        got2 = sess.run(result2)
        got3 = sess.run(result3)
        assert got.shape == (3, 10)
        np.testing.assert_equal(got[0], got[2])
        np.testing.assert_equal(got, got2)
        np.testing.assert_equal(got, got3)
