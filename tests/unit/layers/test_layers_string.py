"""Tests for layers.string"""

import numpy as np
import tensorflow as tf

import deepr as dpr


def test_layers_string_join():
    """Test for StringJoin"""
    layer = dpr.layers.StringJoin()
    result = layer((tf.constant([b"a", b"b"]), tf.constant([0, 1])))
    with tf.Session() as sess:
        got = sess.run(result)
        expected = [b"a 0", b"b 1"]
        np.testing.assert_equal(got, expected)
