"""Tests for common.field"""

import pytest
import tensorflow as tf

import deepr as dpr


@pytest.mark.parametrize("shape, expected", [([None, None], False), ([2], True), ([None, 2], False)])
def test_has_fixed_len(shape, expected):
    """Test has_fixed_len method"""
    field = dpr.Field(name="name", shape=shape, dtype=tf.int32)
    assert field.has_fixed_len() == expected
    assert field.has_fixed_len() != field.has_var_len()


@pytest.mark.parametrize(
    "field",
    [
        dpr.Field(name="name", shape=[None, None], dtype=tf.int32),
        dpr.Field(name="name", shape=[2], dtype=tf.int32),
        dpr.Field(name="name", shape=[None, 2], dtype=tf.int32),
    ],
)
def test_as_feature(field):
    """Test as_feature method"""
    field.as_feature()


def test_startswith():
    """Test startswith method"""
    assert dpr.Field(name="inputPositive", shape=[None, None], dtype=tf.int32).startswith("input")
