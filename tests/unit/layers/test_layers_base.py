# pylint: disable=no-value-for-parameter,invalid-name,unexpected-keyword-arg,redefined-outer-name
"""Tests for layers.base"""

import pytest
import tensorflow as tf

import deepr


@pytest.fixture
def session():
    with tf.Session() as sess:
        yield sess


@deepr.layers.layer(n_in=2, n_out=1)
def Add(tensors):
    """Add"""
    x, y = tensors
    return x + y


@deepr.layers.layer(n_in=2, n_out=2)
def Swap(tensors):
    """Swap"""
    x, y = tensors
    return y, x


@deepr.layers.layer(n_in=1, n_out=1)
def AddOffset(tensors, offset):
    """AddOffset"""
    return tensors + offset


@deepr.layers.layer(n_in=1, n_out=1)
def AddOne(tensors):
    return tensors + 1


def test_layers_call(session):
    """Test layer ability to operate on dictionaries and tuples."""
    offset_layer = AddOffset(inputs="x", outputs="y", offset=1)
    assert isinstance(offset_layer, deepr.layers.Layer)
    result = offset_layer(tf.constant(1))
    result_dict = offset_layer({"x": tf.constant(1)})
    assert session.run(result) == 2
    assert session.run(result_dict) == {"y": 2}


@pytest.mark.parametrize(
    "cls, inputs, outputs, error, inputs_exp, outputs_exp",
    [
        # One input / one output
        (AddOne, None, None, None, "t_0", "t_0"),
        (AddOne, "x", "y", None, "x", "y"),
        (AddOne, ("x", "y"), "y", ValueError, None, None),
        (AddOne, ("x",), "y", ValueError, None, None),
        (AddOne, "x", ("y",), ValueError, None, None),
        (AddOne, "x", ("y", "z"), ValueError, None, None),
        # Two inputs / one output
        (Add, None, None, None, ("t_0", "t_1"), "t_0"),
        (Add, ("x", "y"), "z", None, ("x", "y"), "z"),
        (Add, "x", "z", ValueError, None, None),
        (Add, ("x", "y"), ("y", "z"), ValueError, None, None),
        # Two inputs / two outputs
        (Swap, None, None, None, ("t_0", "t_1"), ("t_0", "t_1")),
        (Swap, ("x", "y"), ("y", "z"), None, ("x", "y"), ("y", "z")),
        (Swap, "x", ("y", "z"), ValueError, None, None),
        (Swap, ("x", "y"), "z", ValueError, None, None),
        (Swap, ("x", "y", "z"), ("y", "z"), ValueError, None, None),
        (Swap, ("x", "y"), ("x", "y", "z"), ValueError, None, None),
    ],
)
def test_layers_inputs_outputs(cls, inputs, outputs, error, inputs_exp, outputs_exp):
    """Test that inputs and outputs are set correctly."""
    if error is not None:
        with pytest.raises(error):
            cls(inputs=inputs, outputs=outputs)
    else:
        instance = cls(inputs=inputs, outputs=outputs)
        assert instance.inputs == inputs_exp
        assert instance.outputs == outputs_exp


def test_layers_decorator_from_forward(session):
    """Test decorator on forward function."""
    # Check decorated function properties
    assert issubclass(AddOffset, deepr.layers.Layer)
    assert AddOffset.__name__ == "AddOffset"
    assert AddOffset.__doc__ == "AddOffset"
    assert AddOffset.__module__ == __name__

    # Check instance properties
    offset_layer = AddOffset(offset=1)
    assert isinstance(offset_layer, AddOffset)
    result = offset_layer.forward(tf.constant(1))
    assert session.run(result) == 2


def test_layers_decorator_from_forward_laziness():
    """Test decorator laziness."""

    @deepr.layers.layer(n_in=1, n_out=1)
    def RaiseForward(tensors):
        raise RuntimeError()

    layer = RaiseForward()

    with pytest.raises(RuntimeError):
        layer.forward(tf.constant(1))


def test_layers_decorator_from_forward_wrong_order():
    """Test wrong order in arguments raises error at decoration time."""
    # pylint: disable=unused-variable
    with pytest.raises(TypeError):

        @deepr.layers.layer(n_in=1, n_out=1)
        def WrongOrder(offset, tensors):
            return tensors + offset


@deepr.layers.layer(n_in=1, n_out=1)
def Identity(tensors) -> tf.data.Dataset:
    return tensors


@deepr.layers.layer(n_in=1, n_out=1)
def Typical(tensors, mode, foo, bar=1):
    # pylint: disable=unused-argument
    return tensors + foo + bar


@pytest.mark.parametrize(
    "cls, args, kwargs, error",
    [
        # Simple layer with no special arguments
        (Identity, (), {}, None),
        (Identity, (), {"tensors": 1}, TypeError),
        (Identity, (1,), None, TypeError),
        (Identity, (), {"foo": 1}, TypeError),
        # Typical layers with positional and keyword arguments
        (Typical, (1,), {}, None),
        (Typical, (1, 2), {}, None),
        (Typical, (), {"foo": 1, "bar": 2}, None),
        (Typical, (1,), {"bar": 2}, None),
        (Typical, (1, 2, 3), {}, TypeError),
        (Typical, (1,), {"foo": 1}, TypeError),
        (Typical, (1,), {"baz": 1}, TypeError),
    ],
)
def test_layers_decorator_instantiation(cls, args, kwargs, error):
    """Test layer instantiation from decorator."""
    if error is not None:
        with pytest.raises(error):
            cls(*args, **kwargs)
    else:
        instance = cls(*args, **kwargs)
        assert isinstance(instance, deepr.layers.Layer)
