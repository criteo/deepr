"""Tests for layers.slice"""

import numpy as np
import tensorflow as tf

import deepr as dpr


def test_layers_slice():
    """Test for Slice"""
    layer = dpr.layers.Slice(begin=1, end=2)
    result = layer(tf.constant([1, 2, 3]))
    with tf.Session() as sess:
        got = sess.run(result)
        np.testing.assert_equal(got, [2])


def test_layers_slice_first():
    """Test for SliceFirst"""
    layer = dpr.layers.SliceFirst(2)
    result = layer(tf.constant([1, 2, 3]))
    with tf.Session() as sess:
        got = sess.run(result)
        np.testing.assert_equal(got, [1, 2])


def test_layers_slice_last():
    """Test for SliceLast"""
    layer = dpr.layers.SliceLast(2)
    result = layer(tf.constant([1, 2, 3]))
    with tf.Session() as sess:
        got = sess.run(result)
        np.testing.assert_equal(got, [2, 3])


def test_layers_slice_last_padded():
    """Test for SliceLastPadded"""
    layer = dpr.layers.SliceLastPadded(padded_value=-1)
    tensor = tf.constant(
        [
            [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
            [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
            [[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]],
        ],
        dtype=tf.float32,
    )
    input_timelines_batch = tf.constant([[0, -1, -1, -1], [2, 4, -1, -1], [1, 1, 0, 4]], dtype=tf.int64)
    result = layer((tensor, input_timelines_batch))
    expected = np.array([[1, 1, 1], [2, 2, 2], [4, 4, 4]], dtype=np.float32)
    with tf.Session() as sess:
        got = sess.run(result)
        np.testing.assert_equal(got, expected)
