"""Tests for layers.deepr"""

import pytest
import tensorflow as tf
import numpy as np

import deepr


def test_layers_sum():
    """Test for Sum"""
    layer = deepr.layers.Sum()
    result = layer((tf.constant(1), tf.constant(2)))
    with tf.Session() as sess:
        assert sess.run(result) == 3


def test_layers_product():
    """Test for Product"""
    layer = deepr.layers.Product()
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

    layer = deepr.layers.DotProduct()
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
    layer = deepr.layers.Dense(16)
    result = layer(tf.ones([8, 8]))
    result2 = layer(tf.ones([8, 8]), reuse=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        got = sess.run(result)
        got2 = sess.run(result2)
        assert got.shape == (8, 16)
        np.testing.assert_equal(got, got2)


def test_layers_dense_index():
    """Test for DenseIndex."""
    x = np.random.random([8, 16])
    indices = np.random.randint(32, size=[8, 4])
    x_tf, indices_tf = tf.constant(x, dtype=tf.float32), tf.constant(indices, dtype=tf.int64)
    kernel = tf.get_variable("kernel", shape=(32, 16))
    bias = tf.get_variable("bias", shape=(32,))
    layer = deepr.layers.DenseIndex(units=32, kernel_name="kernel", bias_name="bias", reuse=True)
    result = layer((x_tf, indices_tf))
    result2 = tf.matmul(x_tf, kernel, transpose_b=True) + tf.expand_dims(bias, axis=0)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        got = sess.run(result)
        got2 = sess.run(result2)
        assert got.shape == (8, 4)
        assert got2.shape == (8, 32)
        for batch in range(8):
            for idx in range(4):
                np.testing.assert_allclose(got[batch, idx], got2[batch, indices[batch, idx]], 1e-4)


def test_layers_conv1d():
    """Test for Conv1d"""
    layer = deepr.layers.Conv1d(filters=5, kernel_size=1)
    result = layer(tf.ones([8, 8, 8]))
    result2 = layer(tf.ones([8, 8, 8]), reuse=True)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        got = sess.run(result)
        got2 = sess.run(result2)
        assert got.shape == (8, 8, 5)
        np.testing.assert_equal(got, got2)


@pytest.mark.parametrize(
    "tensor, mask, expected",
    [
        # normal 1d case
        ([1, 1, 1, 1], [True, True, True, True], [0.25, 0.25, 0.25, 0.25]),
        # normal 2d case
        (
            [[1, 1, 1, 1], [1, 1, 1, 1]],
            [[True, True, True, True], [True, True, True, True]],
            [[0.25, 0.25, 0.25, 0.25], [0.25, 0.25, 0.25, 0.25]],
        ),
        # case with mask
        ([1, 1, 1, 1], [True, True, False, False], [0.5, 0.5, 0, 0]),
        # case with mask
        ([10_000, 0, 0, 0], [True, True, False, False], [1, 0, 0, 0]),
    ],
)
def test_layers_softmax(tensor, mask, expected):
    """Test for Softmax layer"""
    tensor = tf.constant(tensor, dtype=tf.float32)
    mask = tf.constant(mask, dtype=tf.bool)
    expected = np.array(expected, dtype=np.float)
    results = deepr.layers.Softmax()((tensor, mask))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        got = sess.run(results)
        np.testing.assert_equal(expected, got)
