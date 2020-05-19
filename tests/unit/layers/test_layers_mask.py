"""Tests for layers.mask"""

import numpy as np
import tensorflow as tf

import deepr as dpr


def test_layers_mask_equal():
    """Test for Equal"""
    layer = dpr.layers.Equal(values=(0, 1))
    result = layer(tf.constant([0, 1, 2]))
    with tf.Session() as sess:
        got = sess.run(result)
        np.testing.assert_equal(got, [True, True, False])


def test_layers_mask_not_equal():
    """Test for NotEqual"""
    layer = dpr.layers.NotEqual(values=(0, 1))
    result = layer(tf.constant([0, 1, 2]))
    with tf.Session() as sess:
        got = sess.run(result)
        np.testing.assert_equal(got, [False, False, True])


def test_layers_boolean_mask():
    """Test for BooleanMask"""
    layer = dpr.layers.BooleanMask()
    result = layer((tf.constant([0, 1, 2]), tf.constant([True, False, True])))
    with tf.Session() as sess:
        got = sess.run(result)
        np.testing.assert_equal(got, [0, 2])


def test_layers_look_ahead_mask():
    """Test for LookAheadMask"""
    layer = dpr.layers.LookAheadMask()
    tensor = tf.random.uniform((1, 3))
    result = layer(tensor)
    expected = [[0, 1, 1], [0, 0, 1], [0, 0, 0]]
    with tf.Session() as sess:
        got = sess.run(result)
        np.testing.assert_equal(got, expected)


def test_layers_padding_mask():
    """Test for PaddingMask"""
    layer = dpr.layers.PaddingMask(padded_value=-1)
    tensor = [1, 2, 3, -1, -1, 4]
    result = layer(tensor)
    expected = [0, 0, 0, 1, 1, 0]
    with tf.Session() as sess:
        got = sess.run(result)
        np.testing.assert_equal(got, expected)
