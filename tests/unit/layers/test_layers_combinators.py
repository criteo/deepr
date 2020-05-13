# pylint: disable=no-value-for-parameter,invalid-name,unexpected-keyword-arg
"""Tests for layers.combinators"""

import tensorflow as tf

import deepr as dpr


@dpr.layers.layer(n_in=1, n_out=1)
def OffsetLayer(tensors, offset):
    return tensors + offset


@dpr.layers.layer(n_in=2, n_out=1)
def Add(tensors):
    x, y = tensors
    return x + y


def test_layers_sequential():
    """Test basic use of Sequential"""
    layer = dpr.layers.Sequential(OffsetLayer(offset=1, inputs="x"), OffsetLayer(offset=2, outputs="y"))
    result = layer(tf.constant(1))
    result_dict = layer({"x": tf.constant(1)})
    with tf.Session() as sess:
        assert sess.run(result) == 4
        assert sess.run(result_dict)["y"] == 4


def test_layers_sequential_stack():
    """Test use of Sequential with implicit parallelism"""
    layer = dpr.layers.Sequential(dpr.layers.Select(n_in=2), OffsetLayer(offset=2), Add())
    result = layer((tf.constant(2), tf.constant(2)))
    with tf.Session() as sess:
        assert sess.run(result) == 6


def test_layers_sequential_dag():
    """Test use of Sequential as a DAG definition tool"""
    layer = dpr.layers.Sequential(
        OffsetLayer(offset=2, inputs="x", outputs="y"),
        OffsetLayer(offset=2, inputs="x", outputs="z"),
        Add(inputs=("y", "z"), outputs="total"),
    )
    result = layer(tf.constant(1))
    result_dict = layer({"x": tf.constant(1)})
    with tf.Session() as sess:
        assert sess.run(result) == 6
        assert sess.run(result_dict)["total"] == 6


def test_layers_sequential_from_config():
    """Test from_config on sequential"""
    config = {
        "type": "deepr.layers.Sequential",
        "*": [{"type": "deepr.layers.Dense", "units": 10}, {"type": "deepr.layers.Identity"}],
    }
    sequential = dpr.from_config(config)
    assert [type(layer) for layer in sequential.layers] == [dpr.layers.Dense, dpr.layers.Identity]


def test_layers_select():
    """Test basic use of Select"""
    layer = dpr.layers.Select(("x", "y"), "z", n_in=2, indices=1)
    result = layer((tf.constant(1), tf.constant(2)))
    result_dict = layer({"x": tf.constant(1), "y": tf.constant(2)})
    with tf.Session() as sess:
        assert sess.run(result) == 2
        assert sess.run(result_dict)["z"] == 2


def test_layers_select_dag():
    """Test use of Select in a DAG defined with Sequential"""
    layer = dpr.layers.Sequential(
        dpr.layers.Select(("x1", "x2")),
        OffsetLayer(offset=2, inputs="x1", outputs="y1"),
        OffsetLayer(offset=2, inputs="x2", outputs="y2"),
        Add(inputs=("y1", "y2"), outputs="y3"),
        dpr.layers.Select(("y1", "y2", "y3")),
    )
    result = layer((tf.constant(1), tf.constant(2)))
    result_dict = layer({"x1": tf.constant(1), "x2": tf.constant(2)})
    with tf.Session() as sess:
        assert sess.run(result) == (3, 4, 7)
        assert sess.run(result_dict) == {"y1": 3, "y2": 4, "y3": 7}


def test_layers_rename():
    """Test basic use of Rename"""
    add = Add(inputs=("a", "b"), outputs="c")
    layer = dpr.layers.Rename(layer=add, inputs=("x", "y"), outputs="z")
    result = layer((tf.constant(1), tf.constant(1)))
    result_dict = layer({"x": tf.constant(1), "y": tf.constant(1)})
    with tf.Session() as sess:
        assert sess.run(result) == 2
        assert sess.run(result_dict)["z"] == 2


def test_layers_parallel():
    """Test basic use of Parallel"""
    layer1 = Add(inputs=("x1", "x2"), outputs="y1")
    layer2 = OffsetLayer(offset=1, inputs="x3", outputs="y2")
    layer = dpr.layers.Parallel(layer1, layer2)
    result = layer((tf.constant(1), tf.constant(1), tf.constant(2)))
    result_dict = layer({"x1": tf.constant(1), "x2": tf.constant(1), "x3": tf.constant(2)})
    with tf.Session() as sess:
        assert sess.run(result) == (2, 3)
        assert sess.run(result_dict) == {"y1": 2, "y2": 3}
