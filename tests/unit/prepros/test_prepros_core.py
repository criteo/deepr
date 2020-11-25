# pylint: disable=no-value-for-parameter,invalid-name,redefined-outer-name
"""Tests for prepros.deepr"""

import pytest
import numpy as np
import tensorflow as tf

import deepr


@pytest.fixture
def dataset():
    def _gen():
        yield {"a": [0]}
        yield {"a": [0, 1]}

    return tf.data.Dataset.from_generator(_gen, {"a": tf.int32}, {"a": (None,)})


@pytest.mark.parametrize(
    "update, modes, mode, expected",
    [
        (True, None, None, [{"a": [0], "b": [1]}, {"a": [0, 1], "b": [1, 2]}]),
        (False, None, None, [{"b": [1]}, {"b": [1, 2]}]),
        (False, [tf.estimator.ModeKeys.TRAIN], None, [{"b": [1]}, {"b": [1, 2]}]),
        (False, [tf.estimator.ModeKeys.TRAIN], tf.estimator.ModeKeys.TRAIN, [{"b": [1]}, {"b": [1, 2]}]),
        (False, [tf.estimator.ModeKeys.TRAIN], tf.estimator.ModeKeys.EVAL, [{"a": [0]}, {"a": [0, 1]}]),
    ],
)
def test_prepros_map(dataset, update, modes, mode, expected):
    """Test Map behavior with update and modes"""
    prepro_fn = deepr.prepros.Map(lambda x: {"b": x["a"] + 1}, update=update, modes=modes)
    reader = deepr.readers.from_dataset(prepro_fn(dataset, mode))
    np.testing.assert_equal(list(reader), expected)


@pytest.mark.parametrize(
    "modes, mode, expected",
    [
        (None, None, [{"a": [0, 1]}]),
        ([tf.estimator.ModeKeys.TRAIN], None, [{"a": [0, 1]}]),
        ([tf.estimator.ModeKeys.TRAIN], tf.estimator.ModeKeys.TRAIN, [{"a": [0, 1]}]),
        ([tf.estimator.ModeKeys.TRAIN], tf.estimator.ModeKeys.EVAL, [{"a": [0]}, {"a": [0, 1]}]),
    ],
)
def test_prepros_filter(dataset, modes, mode, expected):
    """Test Filter behavior with different modes"""
    prepro_fn = deepr.prepros.Filter(lambda x: {"y": tf.greater(tf.reduce_sum(x["a"]), 0)}, modes=modes)
    reader = deepr.readers.from_dataset(prepro_fn(dataset, mode))
    np.testing.assert_equal(list(reader), expected)


def test_prepros_padded_batch(dataset):
    """Test Padded Batch"""
    fields = [deepr.Field(name="a", shape=[None], dtype=tf.int32, default=-1)]
    prepro_fn = deepr.prepros.PaddedBatch(2, fields)
    reader = deepr.readers.from_dataset(prepro_fn(dataset))
    expected = [{"a": [[0, -1], [0, 1]]}]
    np.testing.assert_equal(list(reader), expected)
