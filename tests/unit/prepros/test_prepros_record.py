# pylint: disable=redefined-outer-name
"""Tests for prepros.record"""

import pytest
import numpy as np
import tensorflow as tf

import deepr


def test_prepros_from_example():
    """Test FromExample."""
    example = tf.train.Example(
        features=tf.train.Features(feature={"x": tf.train.Feature(int64_list=tf.train.Int64List(value=[0, 1, 2, 3]))})
    )
    serialized = example.SerializeToString()
    from_example = deepr.prepros.FromExample(fields=[deepr.Field(name="x", shape=(2, 2), dtype=tf.int64)])
    got = from_example.map_func(serialized)
    assert isinstance(got["x"], tf.Tensor)
    assert got["x"].shape == (2, 2)
    with tf.Session() as sess:
        np.testing.assert_equal(sess.run(got), {"x": np.array([[0, 1], [2, 3]])})


def test_prepros_to_example():
    """Test ToExample."""
    x = deepr.Field(name="x", shape=(2, 2), dtype=tf.int64)
    y = deepr.Field(name="y", shape=(None, None, None), dtype=tf.int64)
    uid = deepr.Field(name="uid", shape=(), dtype=tf.string)
    tensor = {"x": np.array([[0, 1], [2, 3]]), "y": np.ones([2, 3, 4], dtype=np.int64), "uid": b"1234"}
    to_example = deepr.prepros.ToExample(fields=[x, y, uid])
    example = to_example.map_func(tensor)
    assert isinstance(example, tf.Tensor)
    assert example.dtype == tf.string
    assert example.shape == ()
    with tf.Session() as sess:
        sess.run(example)


@pytest.mark.parametrize(
    "field, tensor",
    [
        (deepr.Field(name="x", shape=(), dtype=tf.string), b"1234"),
        (deepr.Field(name="x", shape=(2,), dtype=tf.int64), np.arange(2)),
        (deepr.Field(name="x", shape=(2, 2), dtype=tf.int64), np.reshape(np.arange(2 * 2), (2, 2))),
        (deepr.Field(name="x", shape=(None, 2), dtype=tf.int64), np.reshape(np.arange(2 * 2), (2, 2))),
        (deepr.Field(name="x", shape=(None, None), dtype=tf.int64), np.reshape(np.arange(2 * 2), (2, 2))),
        (deepr.Field(name="x", shape=(2, 3, 4), dtype=tf.int64), np.reshape(np.arange(2 * 3 * 4), (2, 3, 4))),
        (deepr.Field(name="x", shape=(None, 3, 4), dtype=tf.int64), np.reshape(np.arange(2 * 3 * 4), (2, 3, 4))),
        (deepr.Field(name="x", shape=(None, None, 4), dtype=tf.int64), np.reshape(np.arange(2 * 3 * 4), (2, 3, 4))),
        (deepr.Field(name="x", shape=(None, None, None), dtype=tf.int64), np.reshape(np.arange(2 * 3 * 4), (2, 3, 4))),
    ],
)
def test_end_to_end(field, tensor):
    """Test end-to-end, serialize then parse."""
    to_example = deepr.prepros.ToExample(fields=[field])
    from_example = deepr.prepros.FromExample(fields=[field])
    example = to_example.map_func({field.name: tensor})
    got = from_example.map_func(example)[field.name]
    if field.is_sparse():
        got = deepr.layers.ToDense(default_value=field.default)(got)
    with tf.Session() as sess:
        np.testing.assert_equal(sess.run(got), tensor)
