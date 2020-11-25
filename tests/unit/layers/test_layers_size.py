"""Tests for layers.size"""

import tensorflow as tf

import deepr


def test_layers_is_min_size():
    """Test for IsMinSize"""
    layer = deepr.layers.IsMinSize(2)
    result = layer(tf.ones([2, 3]))
    with tf.Session() as sess:
        assert sess.run(result)
