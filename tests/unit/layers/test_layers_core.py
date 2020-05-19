"""Tests for layers.deepr"""

import pytest
import tensorflow as tf
import numpy as np

import deepr as dpr


def test_layers_sum():
    """Test for Sum"""
    layer = dpr.layers.Sum()
    result = layer((tf.constant(1), tf.constant(2)))
    with tf.Session() as sess:
        assert sess.run(result) == 3


def test_layers_product():
    """Test for Product"""
    layer = dpr.layers.Product()
    result = layer((tf.constant(1), tf.constant(2)))
    with tf.Session() as sess:
        assert sess.run(result) == 2


@pytest.mark.parametrize(
    "left_dim, right_dim, expected_dim",
    [
        ([2], [2], []),
        ([2, 3, 5], [2, 5], [2, 3]),
        ([2, 3, 4, 5], [2, 5], [2, 3, 4]),
        ([2, 3, 5], [2, 3, 5], [2, 3]),
        ([2, 3, 4, 5], [2, 3, 5], [2, 3, 4]),
    ],
)
def test_layers_dot_product(left_dim, right_dim, expected_dim):
    """Test for DotProduct"""

    def _naive_dot_product(left, right):
        for _ in range(len(left.shape) - len(right.shape)):
            right = tf.expand_dims(right, axis=-2)
        return tf.reduce_sum(left * right, axis=-1)

    layer = dpr.layers.DotProduct()
    left = tf.constant(np.random.random(left_dim))
    right = tf.constant(np.random.random(right_dim))
    got_tf = layer((left, right))
    naive_tf = _naive_dot_product(left, right)

    with tf.Session() as sess:
        got = sess.run(got_tf)
        naive = sess.run(naive_tf)
        assert list(got.shape) == expected_dim
        np.testing.assert_almost_equal(got, naive, decimal=7)


def test_layers_dense():
    """Test for Dense"""
    layer = dpr.layers.Dense(16)
    result = layer(tf.ones([8, 8]))
    result2 = layer(tf.ones([8, 8]), reuse=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        got = sess.run(result)
        got2 = sess.run(result2)
        assert got.shape == (8, 16)
        np.testing.assert_equal(got, got2)


def test_layers_conv1d():
    """Test for Conv1d"""
    layer = dpr.layers.Conv1d(filters=5, kernel_size=1)
    result = layer(tf.ones([8, 8, 8]))
    result2 = layer(tf.ones([8, 8, 8]), reuse=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        got = sess.run(result)
        got2 = sess.run(result2)
        assert got.shape == (8, 8, 5)
        np.testing.assert_equal(got, got2)
