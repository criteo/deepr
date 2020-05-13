# pylint: disable=no-value-for-parameter,invalid-name,unexpected-keyword-arg
"""Tests for prepros.base"""

import pytest
import tensorflow as tf
import numpy as np

import deepr as dpr


def gen():
    yield {"a": [0]}
    yield {"a": [0, 1]}


def test_prepros_decorator_from_apply():
    """Create preprocessor from an apply function"""

    @dpr.prepros.prepro
    def AddOffset(dataset, offset):
        return dataset.map(lambda x: {"b": x["a"] + offset})

    ds = tf.data.Dataset.from_generator(gen, {"a": tf.int32}, {"a": (None,)})

    # Positional argument
    add_one = AddOffset(1)
    reader = dpr.readers.from_dataset(add_one(ds))
    expected = [{"b": [1]}, {"b": [1, 2]}]
    np.testing.assert_equal(list(reader), expected)

    # Keyword argument
    add_one = AddOffset(offset=1)
    reader = dpr.readers.from_dataset(add_one(ds))
    expected = [{"b": [1]}, {"b": [1, 2]}]
    np.testing.assert_equal(list(reader), expected)


def test_prepros_decorator_from_constructor():
    """Create preprocessor from a Prepro constructor"""

    @dpr.prepros.prepro
    def AddOffset(offset):
        return dpr.prepros.Map(lambda x: {"b": x["a"] + offset}, update=False)

    ds = tf.data.Dataset.from_generator(gen, {"a": tf.int32}, {"a": (None,)})

    # Positional argument
    add_one = AddOffset(1)
    reader = dpr.readers.from_dataset(add_one(ds))
    expected = [{"b": [1]}, {"b": [1, 2]}]
    np.testing.assert_equal(list(reader), expected)

    # Keyword argument
    add_one = AddOffset(offset=1)
    reader = dpr.readers.from_dataset(add_one(ds))
    expected = [{"b": [1]}, {"b": [1, 2]}]
    np.testing.assert_equal(list(reader), expected)


def test_prepros_decorator_signatures():
    """Test incorrect use of decorator"""
    # pylint: disable=unused-argument,unused-variable,too-many-function-args

    # Simple preprocessor with no arguments
    @dpr.prepros.prepro
    def Simple(dataset) -> tf.data.Dataset:
        pass

    Simple()

    with pytest.raises(TypeError):
        Simple(dataset=1)

    with pytest.raises(TypeError):
        Simple(1)

    with pytest.raises(TypeError):
        Simple(foo=1)

    # Typical preprocessor with positional and keyword arguments
    @dpr.prepros.prepro
    def Typical(dataset, mode, foo, bar=1) -> tf.data.Dataset:
        pass

    Typical(1)
    Typical(1, 2)
    Typical(foo=1, bar=2)
    Typical(1, bar=2)

    with pytest.raises(TypeError):
        Typical(1, 2, 3)

    with pytest.raises(TypeError):
        Typical(1, foo=1)

    with pytest.raises(TypeError):
        Typical(1, baz=1)

    # From constructor
    def FromConstructor(foo, bar=1) -> dpr.prepros.Prepro:
        pass

    FromConstructor(1, 2)
    FromConstructor(1, bar=2)
    FromConstructor(foo=1, bar=2)

    with pytest.raises(TypeError):
        FromConstructor(1, 2, 3)

    with pytest.raises(TypeError):
        Typical(1, foo=1)

    with pytest.raises(TypeError):
        Typical(1, baz=1)

    # Test wrong order in arguments raises error at decoration time
    with pytest.raises(TypeError):

        @dpr.prepros.prepro
        def WrongOrder(offset, dataset):
            pass


def test_prepros_decorator_laziness_from_apply():
    """Laziness is especially useful if custom prepro use hash tables"""

    @dpr.prepros.prepro
    def ErrorPrepro(dataset):
        raise ValueError()

    prepro_fn = ErrorPrepro()
    with pytest.raises(ValueError):
        prepro_fn(None)


def test_prepros_decorator_laziness_from_constructor():
    """Laziness is especially useful if custom prepro use hash tables"""

    @dpr.prepros.prepro
    def ErrorPrepro():
        raise ValueError()

    prepro_fn = ErrorPrepro()
    with pytest.raises(ValueError):
        prepro_fn(None)
