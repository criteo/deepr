# pylint: disable=no-value-for-parameter,invalid-name,unexpected-keyword-arg
"""Tests for layers.combinators"""

import pytest
import tensorflow as tf

import deepr


@deepr.layers.layer(n_in=1, n_out=1)
def OffsetLayer(tensors, offset):
    return tensors + offset


@deepr.layers.layer(n_in=2, n_out=1)
def Add(tensors):
    x, y = tensors
    return x + y


def test_layers_sequential_dag_legacy():
    """Test legacy class name."""
    assert deepr.layers.DAG == deepr.layers.Sequential


def test_layers_dag():
    """Test basic use of DAG"""
    layer = deepr.layers.DAG(OffsetLayer(offset=1, inputs="x"), OffsetLayer(offset=2, outputs="y"))
    result = layer(tf.constant(1))
    result_dict = layer({"x": tf.constant(1)})
    with tf.Session() as sess:
        assert sess.run(result) == 4
        assert sess.run(result_dict)["y"] == 4


def test_layers_dag_stack():
    """Test use of DAG with implicit parallelism"""
    layer = deepr.layers.DAG(deepr.layers.Select(n_in=2), OffsetLayer(offset=2), Add())
    result = layer((tf.constant(2), tf.constant(2)))
    with tf.Session() as sess:
        assert sess.run(result) == 6


def test_layers_dag_dag():
    """Test use of DAG as a DAG definition tool"""
    layer = deepr.layers.DAG(
        OffsetLayer(offset=2, inputs="x", outputs="y"),
        OffsetLayer(offset=2, inputs="x", outputs="z"),
        Add(inputs=("y", "z"), outputs="total"),
    )
    result = layer(tf.constant(1))
    result_dict = layer({"x": tf.constant(1)})
    with tf.Session() as sess:
        assert sess.run(result) == 6
        assert sess.run(result_dict)["total"] == 6


def test_layers_dag_from_config():
    """Test from_config on dag"""
    config = {
        "type": "deepr.layers.DAG",
        "*": [{"type": "deepr.layers.Dense", "units": 10}, {"type": "deepr.layers.Identity"}],
    }
    dag = deepr.from_config(config)
    assert [type(layer) for layer in dag.layers] == [deepr.layers.Dense, deepr.layers.Identity]


def test_layers_select():
    """Test basic use of Select"""
    layer = deepr.layers.Select(("x", "y"), "z", n_in=2, indices=1)
    result = layer((tf.constant(1), tf.constant(2)))
    result_dict = layer({"x": tf.constant(1), "y": tf.constant(2)})
    with tf.Session() as sess:
        assert sess.run(result) == 2
        assert sess.run(result_dict)["z"] == 2


@pytest.mark.parametrize(
    "inputs, outputs, indices, expected",
    [
        ("foo", None, None, deepr.layers.Select(n_in=1, inputs="foo", outputs="foo", indices=0)),
        (
            ("foo", "bar"),
            None,
            None,
            deepr.layers.Select(n_in=2, inputs=("foo", "bar"), outputs=("foo", "bar"), indices=(0, 1)),
        ),
        (("foo", "bar"), None, 0, deepr.layers.Select(n_in=2, inputs=("foo", "bar"), outputs="foo", indices=0)),
        (("foo", "bar"), None, 1, deepr.layers.Select(n_in=2, inputs=("foo", "bar"), outputs="bar", indices=1)),
        (
            ("foo", "bar"),
            None,
            (1, 0),
            deepr.layers.Select(n_in=2, inputs=("foo", "bar"), outputs=("bar", "foo"), indices=(1, 0)),
        ),
    ],
)
def test_layers_select_init(inputs, outputs, indices, expected):
    got = deepr.layers.Select(inputs=inputs, outputs=outputs, indices=indices)
    assert got.inputs == expected.inputs
    assert got.outputs == expected.outputs
    assert got.n_in == expected.n_in
    assert got.indices == expected.indices


def test_layers_select_dag():
    """Test use of Select in a DAG defined with DAG"""
    layer = deepr.layers.DAG(
        deepr.layers.Select(("x1", "x2")),
        OffsetLayer(offset=2, inputs="x1", outputs="y1"),
        OffsetLayer(offset=2, inputs="x2", outputs="y2"),
        Add(inputs=("y1", "y2"), outputs="y3"),
        deepr.layers.Select(("y1", "y2", "y3")),
    )
    result = layer((tf.constant(1), tf.constant(2)))
    result_dict = layer({"x1": tf.constant(1), "x2": tf.constant(2)})
    with tf.Session() as sess:
        assert sess.run(result) == (3, 4, 7)
        assert sess.run(result_dict) == {"y1": 3, "y2": 4, "y3": 7}


def test_layers_rename():
    """Test basic use of Rename"""
    add = Add(inputs=("a", "b"), outputs="c")
    layer = deepr.layers.Rename(layer=add, inputs=("x", "y"), outputs="z")
    result = layer((tf.constant(1), tf.constant(1)))
    result_dict = layer({"x": tf.constant(1), "y": tf.constant(1)})
    with tf.Session() as sess:
        assert sess.run(result) == 2
        assert sess.run(result_dict)["z"] == 2


def test_layers_parallel():
    """Test basic use of Parallel"""
    layer1 = Add(inputs=("x1", "x2"), outputs="y1")
    layer2 = OffsetLayer(offset=1, inputs="x3", outputs="y2")
    layer = deepr.layers.Parallel(layer1, layer2)
    result = layer((tf.constant(1), tf.constant(1), tf.constant(2)))
    result_dict = layer({"x1": tf.constant(1), "x2": tf.constant(1), "x3": tf.constant(2)})
    with tf.Session() as sess:
        assert sess.run(result) == (2, 3)
        assert sess.run(result_dict) == {"y1": 2, "y2": 3}
