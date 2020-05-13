# pylint: disable=no-value-for-parameter,invalid-name,unexpected-keyword-arg
"""Tests for layers.base"""

import pytest
import tensorflow as tf

import deepr as dpr


@dpr.layers.layer(n_in=2, n_out=1)
def Add(tensors):
    x, y = tensors
    return x + y


@dpr.layers.layer(n_in=2, n_out=2)
def Swap(tensors):
    x, y = tensors
    return y, x


@dpr.layers.layer(n_in=1, n_out=1)
def AddOffset(tensors, offset):
    return tensors + offset


@dpr.layers.layer(n_in=1, n_out=1)
def AddOne():
    return AddOffset(offset=1)


@pytest.mark.parametrize(
    "cls, inputs, outputs, error, inputs_exp, outputs_exp",
    [
        (AddOne, None, None, False, "t_0", "t_0"),
        (AddOne, "x", "y", False, "x", "y"),
        (AddOne, ("x", "y"), "y", True, None, None),
        (AddOne, ("x",), "y", True, None, None),
        (AddOne, "x", ("y",), True, None, None),
        (AddOne, "x", ("y", "z"), True, None, None),
        (Add, None, None, False, ("t_0", "t_1"), "t_0"),
        (Add, ("x", "y"), "z", False, ("x", "y"), "z"),
        (Add, "x", "z", True, None, None),
        (Add, ("x", "y"), ("y", "z"), True, None, None),
        (Swap, None, None, False, ("t_0", "t_1"), ("t_0", "t_1")),
        (Swap, ("x", "y"), ("y", "z"), False, ("x", "y"), ("y", "z")),
        (Swap, "x", ("y", "z"), True, None, None),
        (Swap, ("x", "y"), "z", True, None, None),
        (Swap, ("x", "y", "z"), ("y", "z"), True, None, None),
        (Swap, ("x", "y"), ("x", "y", "z"), True, None, None),
    ],
)
def test_layers_inputs_outputs(cls, inputs, outputs, error, inputs_exp, outputs_exp):
    """Test that inputs / outputs are set correctly"""
    if error:
        with pytest.raises(ValueError):
            cls(inputs=inputs, outputs=outputs)

    else:
        instance = cls(inputs=inputs, outputs=outputs)
        assert instance.inputs == inputs_exp
        assert instance.outputs == outputs_exp


def test_layers_call():
    """Test `Layer.__call__` ability to operate on dict / tuples"""
    offset_layer = AddOffset(offset=1, inputs="x", outputs="y")
    assert isinstance(offset_layer, dpr.layers.Layer)
    result = offset_layer(tf.constant(1))
    result_dict = offset_layer({"x": tf.constant(1)})
    with tf.Session() as sess:
        assert sess.run(result) == 2
        assert sess.run(result_dict) == {"y": 2}


def test_layers_decorator_from_forward():
    """Test decorator on forward function"""

    @dpr.layers.layer(n_in=1, n_out=1)
    def MyOffset(tensors, foo, bar=1):
        return tensors + foo + 2 * bar

    # Positional arguments
    offset_layer = MyOffset(1, 2)
    assert isinstance(offset_layer, dpr.layers.Layer)
    result = offset_layer.forward(tf.constant(1))
    with tf.Session() as sess:
        assert sess.run(result) == 6

    # Keyword arguments
    offset_layer = MyOffset(foo=1, bar=2)
    assert isinstance(offset_layer, dpr.layers.Layer)
    result = offset_layer.forward(tf.constant(1))
    with tf.Session() as sess:
        assert sess.run(result) == 6


def test_layers_decorator_from_constructor():
    """Test decorator on constructors"""

    @dpr.layers.layer(n_in=1, n_out=1)
    def MyOffset(foo, bar=1):
        return AddOffset(foo + 2 * bar)

    # Positional arguments
    offset_layer = MyOffset(1, 2)
    assert isinstance(offset_layer, dpr.layers.Layer)
    result = offset_layer.forward(tf.constant(1))
    with tf.Session() as sess:
        assert sess.run(result) == 6

    # Keyword arguments
    offset_layer = MyOffset(foo=1, bar=2)
    assert isinstance(offset_layer, dpr.layers.Layer)
    result = offset_layer.forward(tf.constant(1))
    with tf.Session() as sess:
        assert sess.run(result) == 6


def test_layers_decorator_signatures():
    """Test incorrect use of decorator"""
    # pylint: disable=unused-argument,unused-variable,too-many-function-args

    # Simple Layer with no arguments
    @dpr.layers.layer(n_in=1, n_out=1)
    def Simple(tensors) -> tf.data.Dataset:
        pass

    Simple()

    with pytest.raises(TypeError):
        Simple(tensors=1)

    with pytest.raises(TypeError):
        Simple(1)

    with pytest.raises(TypeError):
        Simple(foo=1)

    # Typical Layer with positional and keyword arguments
    @dpr.layers.layer(n_in=1, n_out=1)
    def Typical(tensors, mode, foo, bar=1):
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
    @dpr.layers.layer(n_in=1, n_out=1)
    def FromConstructor(foo, bar=1) -> dpr.layers.Layer:
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

        @dpr.layers.layer(n_in=1, n_out=1)
        def WrongOrder(offset, tensors):
            pass


def test_layers_decorator_laziness_from_forward():
    """Test decorator laziness and keyword arguments check"""

    @dpr.layers.layer(n_in=1, n_out=1)
    def Offset(tensors, offset):
        raise RuntimeError()

    Offset(offset=1)  # Test laziness

    with pytest.raises(TypeError):
        Offset(baz=1)

    with pytest.raises(TypeError):
        Offset(tensors=1)

    with pytest.raises(TypeError):
        Offset()


def test_layers_decorator_laziness_from_constructor():
    """Test decorator laziness and keyword arguments check"""

    @dpr.layers.layer(n_in=1, n_out=1)
    def Offset(offset):
        raise RuntimeError()

    Offset(offset=1)  # Test laziness

    with pytest.raises(TypeError):
        Offset(baz=1)

    with pytest.raises(TypeError):
        Offset(tensors=1)

    with pytest.raises(TypeError):
        Offset()
