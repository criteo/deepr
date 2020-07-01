"""Tensorflow Graph utilities."""

import logging
from typing import Dict, List

import tensorflow as tf
from tensorflow.python.platform import gfile


LOGGER = logging.getLogger(__name__)


INIT_ALL_TABLES = "init_all_tables"


def import_graph_def(path_pb: str, name: str = ""):
    """Import Graph Definition from protobuff into the current Graph.

    Parameters
    ----------
    path_pb : str
        Path to .pb file
    """
    with gfile.FastGFile(f"{path_pb}", "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def, name=name)


def get_by_name(graph: tf.Graph, name: str):
    """Return op in Graph with name or None if not found.

    Parameters
    ----------
    graph : tf.Graph
        A Tensorflow Graph

    Returns
    -------
    tf.Operation or None
    """
    for node in graph.as_graph_def().node:
        if node.name == name:
            return graph.as_graph_element(node.name)
    return None


def get_feedable_tensors(graph: tf.Graph, names: List[str]) -> Dict[str, tf.Tensor]:
    """Retrieve feed tensors from graph.

    Parameters
    ----------
    graph : tf.Graph
        A Tensorflow Graph
    names : List[str]
        List of operations or tensor names.

    Returns
    -------
    Dict[str, tf.Tensor]
        Mapping of names to tf.Tensor
    """
    feedable_tensors = {}
    for name in names:
        op_or_tensor = graph.as_graph_element(name)
        if isinstance(op_or_tensor, tf.Tensor):
            tensor = op_or_tensor
        else:
            if len(op_or_tensor.outputs) > 1:
                raise ValueError(f"Found more than one tensor for operation {op_or_tensor}")
            tensor = op_or_tensor.outputs[0]
        if not graph.is_feedable(tensor):
            raise ValueError(f"{name} should be feedable but is not")
        feedable_tensors[name] = tensor
    return feedable_tensors


def get_fetchable_tensors(graph: tf.Graph, names: List[str]) -> Dict[str, tf.Tensor]:
    """Retrieve fetch tensors from graph.

    Parameters
    ----------
    graph : tf.Graph
        A Tensorflow Graph
    names : List[str]
        List of operations or tensor names

    Returns
    -------
    Dict[str, tf.Tensor]
        Mapping of names to tf.Tensor
    """
    fetchable_tensors = {}
    for name in names:
        op_or_tensor = graph.as_graph_element(name)
        if isinstance(op_or_tensor, tf.Tensor):
            tensor = op_or_tensor
        else:
            if len(op_or_tensor.outputs) > 1:
                raise ValueError(f"Found more than one tensor for operation {op_or_tensor}")
            tensor = op_or_tensor.outputs[0]
        if not graph.is_fetchable(tensor):
            raise ValueError(f"{name} should be fetchable but is not")
        fetchable_tensors[name] = tensor
    return fetchable_tensors
