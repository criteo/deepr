# pylint: disable=redefined-outer-name,no-value-for-parameter
"""Tests for layers.transformer"""

import pytest
import numpy as np
import tensorflow as tf

import deepr


@pytest.fixture
def session():
    with tf.Session() as sess:
        yield sess


def test_layers_transformer(session):
    """Test for layers.transformer"""
    batch_size, sequence_length, dim = 8, 16, 32
    transformer = deepr.layers.Transformer(dim=dim)
    input_positive = tf.constant(np.random.random(size=[batch_size, sequence_length, dim]), dtype=tf.float32)
    input_mask = np.random.choice([True, False], size=[batch_size, sequence_length])
    user_embeddings = transformer((input_positive, input_mask), deepr.TRAIN)
    session.run(tf.global_variables_initializer())
    session.run(user_embeddings)


def attention_mask_np(t):
    batch_size, sequence_length = t.shape
    m = np.zeros([batch_size, sequence_length, sequence_length], dtype=np.bool)
    for b in range(batch_size):
        for i in range(sequence_length):
            for j in range(sequence_length):
                if t[b, i] and t[b, j] and j <= i:
                    m[b, i, j] = True
    return m


def test_layers_attention_mask(session):
    """Test for layers.AttentionMask."""
    batch_size, sequence_length = 8, 16
    layer = deepr.layers.AttentionMask(use_look_ahead_mask=True)
    input_mask = np.random.choice([True, False], size=[batch_size, sequence_length])
    got = session.run(layer(input_mask))
    expected = attention_mask_np(input_mask)
    np.testing.assert_equal(got, expected)


def test_layers_self_multiheadattention(session):
    """Test for layers.SelfMultiheadAttention."""
    batch_size, sequence_length, num_heads, dim_head = 8, 16, 4, 8
    layer = deepr.layers.SelfMultiheadAttention(num_heads=num_heads, dim_head=dim_head, residual_connection=False)
    x = tf.constant(np.random.random(size=[batch_size, num_heads, sequence_length, dim_head]), dtype=tf.float32)
    mask = np.random.choice([True, False], size=[batch_size, sequence_length])
    attn_mask = attention_mask_np(mask)
    got = session.run(layer.scaled_dot_attention(x, x, x, attn_mask))
    assert got.shape == (batch_size, num_heads, sequence_length, dim_head)
    for b in range(batch_size):
        for i in range(sequence_length):
            if not mask[b, i]:
                assert got[b, :, i].sum() == 0
