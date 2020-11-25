# pylint: disable=not-callable,no-value-for-parameter
"""Test for layers.LSTM."""

import numpy as np
import tensorflow as tf

import pytest

import deepr


@pytest.mark.parametrize("bidirectional", [False, True])
def test_layers_lstm(bidirectional):
    """Test for layers.LSTM"""
    batch, nwords, dim, units = 10, 8, 4, 16
    with tf.Session() as sess:
        words_tf = tf.constant(np.random.random([batch, nwords, dim]), dtype=tf.float32)
        nwords_tf = tf.constant(np.random.randint(8, size=[batch]), dtype=tf.int32)
        layer = deepr.layers.LSTM(units, bidirectional=bidirectional)
        outputs_tf, hidden_tf, output_tf = layer((words_tf, nwords_tf))
        sess.run(tf.global_variables_initializer())
        outputs, hidden, output = sess.run((outputs_tf, hidden_tf, output_tf))
        assert outputs.shape == (batch, nwords, units * 2 if bidirectional else units)
        assert hidden.shape == (batch, units * 2 if bidirectional else units)
        assert output.shape == (batch, units * 2 if bidirectional else units)
