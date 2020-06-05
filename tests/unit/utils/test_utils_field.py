"""Tests for common.field"""

import pytest
import tensorflow as tf

import deepr as dpr


@pytest.mark.parametrize(
    "field, expected",
    [
        (dpr.Field(name="name", shape=[None, None], dtype=tf.int64), tf.io.VarLenFeature(dtype=tf.int64)),
        (dpr.Field(name="name", shape=[None], dtype=tf.int64), tf.io.FixedLenSequenceFeature(shape=(), dtype=tf.int64)),
        (dpr.Field(name="name", shape=[2], dtype=tf.int64), tf.io.FixedLenFeature(shape=(2,), dtype=tf.int64)),
        (
            dpr.Field(name="name", shape=[None, 2], dtype=tf.int64),
            tf.io.FixedLenSequenceFeature(shape=(2,), dtype=tf.int64),
        ),
    ],
)
def test_feature_specs(field, expected):
    """Test as_feature method"""
    assert field.feature_specs == expected


def test_startswith():
    """Test startswith method"""
    assert dpr.Field(name="inputPositive", shape=[None, None], dtype=tf.int32).startswith("input")
