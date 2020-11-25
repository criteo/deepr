"""Tests for layers.slice"""

import numpy as np
import tensorflow as tf

import deepr


def test_layers_slice():
    """Test for Slice"""
    layer = deepr.layers.Slice(begin=1, end=2)
    result = layer(tf.constant([1, 2, 3]))
    with tf.Session() as sess:
        got = sess.run(result)
        np.testing.assert_equal(got, [2])


def test_layers_slice_first():
    """Test for SliceFirst"""
    layer = deepr.layers.SliceFirst(2)
    result = layer(tf.constant([1, 2, 3]))
    with tf.Session() as sess:
        got = sess.run(result)
        np.testing.assert_equal(got, [1, 2])


def test_layers_slice_last():
    """Test for SliceLast"""
    layer = deepr.layers.SliceLast(2)
    result = layer(tf.constant([1, 2, 3]))
    with tf.Session() as sess:
        got = sess.run(result)
        np.testing.assert_equal(got, [2, 3])


def test_layers_slice_last_padded():
    """Test for SliceLastPadded"""
    layer = deepr.layers.SliceLastPadded()
    tensor = tf.constant(
        [
            [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
            [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
            [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
        ],
        dtype=tf.float32,
    )
    mask = tf.constant(
        [[True, False, False, False], [True, True, False, False], [True, False, True, True]], dtype=tf.bool
    )
    result = layer((tensor, mask))
    expected = np.array([[1, 1, 1], [2, 2, 2], [4, 4, 4]], dtype=np.float32)
    with tf.Session() as sess:
        got = sess.run(result)
        np.testing.assert_equal(got, expected)
