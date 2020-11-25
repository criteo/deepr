"""Tests for layers.mask"""

import numpy as np
import tensorflow as tf

import deepr


def test_layers_mask_equal():
    """Test for Equal"""
    layer = deepr.layers.Equal(values=(0, 1))
    result = layer(tf.constant([0, 1, 2]))
    with tf.Session() as sess:
        got = sess.run(result)
        np.testing.assert_equal(got, [True, True, False])


def test_layers_mask_not_equal():
    """Test for NotEqual"""
    layer = deepr.layers.NotEqual(values=(0, 1))
    result = layer(tf.constant([0, 1, 2]))
    with tf.Session() as sess:
        got = sess.run(result)
        np.testing.assert_equal(got, [False, False, True])


def test_layers_boolean_mask():
    """Test for BooleanMask"""
    layer = deepr.layers.BooleanMask()
    result = layer((tf.constant([0, 1, 2]), tf.constant([True, False, True])))
    with tf.Session() as sess:
        got = sess.run(result)
        np.testing.assert_equal(got, [0, 2])
