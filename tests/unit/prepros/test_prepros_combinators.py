# pylint: disable=no-value-for-parameter,invalid-name
"""Tests for prepros.combinators"""

import numpy as np
import tensorflow as tf

import deepr as dpr

from deepr.prepros.combinators import _FusedFilter, _FusedMap


def test_prepros_fused_map():
    """Test SerialMap"""
    prepro_fn = _FusedMap(
        dpr.prepros.Map(dpr.layers.Sum(inputs=("a", "b"), outputs="c")),
        dpr.prepros.Map(dpr.layers.Sum(inputs=("a", "c"), outputs="d")),
    )

    def gen():
        yield {"a": 1, "b": 2}

    dataset = tf.data.Dataset.from_generator(gen, {"a": tf.int32, "b": tf.int32}, {"a": (), "b": ()})
    reader = dpr.readers.from_dataset(prepro_fn(dataset))
    expected = [{"a": 1, "b": 2, "c": 3, "d": 4}]
    np.testing.assert_equal(list(reader), expected)


def test_prepros_fused_filter():
    """Test SerialFilter"""
    prepro_fn = _FusedFilter(
        dpr.prepros.Filter(dpr.layers.IsMinSize(inputs="a", outputs="a_size", size=2)),
        dpr.prepros.Filter(dpr.layers.IsMinSize(inputs="b", outputs="b_size", size=2)),
    )

    def gen():
        yield {"a": [0], "b": [0, 1]}
        yield {"a": [0, 1], "b": [0]}
        yield {"a": [0, 1], "b": [0, 1]}

    dataset = tf.data.Dataset.from_generator(gen, {"a": tf.int32, "b": tf.int32}, {"a": (None,), "b": (None,)})
    reader = dpr.readers.from_dataset(prepro_fn(dataset))
    expected = [{"a": [0, 1], "b": [0, 1]}]
    np.testing.assert_equal(list(reader), expected)


def test_prepros_serial():
    """Test Serial"""
    # pylint: disable=protected-access

    def DummyFactory():
        return dpr.prepros.Serial(dpr.prepros.Filter(dpr.layers.IsMinSize(inputs="b", outputs="b_size", size=2)))

    prepro_fn = dpr.prepros.Serial(
        [
            dpr.prepros.Map(dpr.layers.Sum(inputs=("a", "b"), outputs="c")),
            dpr.prepros.Filter(dpr.layers.IsMinSize(inputs="a", outputs="a_size", size=2)),
            dpr.prepros.Serial(DummyFactory()),
        ]
    )
    assert len(prepro_fn._preprocessors) == 2

    def gen():
        yield {"a": [0], "b": [0, 1]}
        yield {"a": [0, 1], "b": [0]}
        yield {"a": [0, 1], "b": [0, 1]}

    dataset = tf.data.Dataset.from_generator(gen, {"a": tf.int32, "b": tf.int32}, {"a": (None,), "b": (None,)})
    reader = dpr.readers.from_dataset(prepro_fn(dataset))
    expected = [{"a": [0, 1], "b": [0, 1], "c": [0, 2]}]
    np.testing.assert_equal(list(reader), expected)


def test_prepros_serial_from_config():
    """Test from_config on Serial"""
    config = {
        "type": "deepr.prepros.Serial",
        "*": [{"type": "deepr.prepros.Repeat", "count": 1}, {"type": "deepr.prepros.Batch", "batch_size": 32}],
    }
    serial = dpr.from_config(config)
    assert [type(prepro) for prepro in serial.preprocessors] == [dpr.prepros.Repeat, dpr.prepros.Batch]
