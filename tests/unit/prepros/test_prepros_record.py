# pylint: disable=redefined-outer-name
"""Tests for prepros.record"""

import numpy as np
import tensorflow as tf

import deepr as dpr


def test_prepros_from_example():
    """Test FromExample."""
    example = tf.train.Example(
        features=tf.train.Features(feature={"x": tf.train.Feature(int64_list=tf.train.Int64List(value=[0, 1, 2, 3]))})
    )
    serialized = example.SerializeToString()
    from_example = dpr.prepros.FromExample(fields=[dpr.Field(name="x", shape=(2, 2), dtype=tf.int64)])
    got = from_example.parse_fn(serialized)
    assert isinstance(got["x"], tf.Tensor)
    assert got["x"].shape == (2, 2)
    with tf.Session() as sess:
        np.testing.assert_equal(sess.run(got), {"x": np.array([[0, 1], [2, 3]])})


def test_prepros_to_example():
    """Test ToExample."""
    tensor = {"x": np.array([[0, 1], [2, 3]])}
    to_example = dpr.prepros.ToExample(fields=[dpr.Field(name="x", shape=(2, 2), dtype=tf.int64)])
    example = to_example.serialize_fn(tensor)
    assert isinstance(example, tf.Tensor)
    assert example.dtype == tf.string
    assert example.shape == ()


def test_end_to_end():
    """Test end-to-end, serialize then parse."""
    field = dpr.Field(name="x", shape=(2, 2), dtype=tf.int64)
    tensor = {"x": np.array([[0, 1], [2, 3]])}
    to_example = dpr.prepros.ToExample(fields=[field])
    from_example = dpr.prepros.FromExample(fields=[field])
    example = to_example.serialize_fn(tensor)
    got = from_example.parse_fn(example)
    with tf.Session() as sess:
        np.testing.assert_equal(sess.run(got), tensor)
