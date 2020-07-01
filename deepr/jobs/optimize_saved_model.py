"""Converts SavedModel into an optimized protobuf for inference"""

from dataclasses import dataclass, field
import logging
import re
from typing import List, Dict, Iterable, Union

import tensorflow as tf
from tensorflow.python.tools.freeze_graph import freeze_graph_with_def_protos
from tensorflow.python.framework.graph_util import extract_sub_graph

from deepr.io import Path
from deepr.jobs import base
from deepr.utils.graph import INIT_ALL_TABLES


LOGGER = logging.getLogger(__name__)


class TensorsNotFoundError(ValueError):
    """No tensors found for those names.

    Most Tensorflow operators take a ``name`` argument that should be
    used if you want to use a custom name for a tensor, otherwise a
    default will be used.

    If accessing the operator creating a Tensor is not possible, you
    can use ``tf.identity`` to name the Tensor. However, note that it
    adds a new op to the graph that creates a copy of the input Tensor,
    and thus should be limited to avoid overhead.
    """

    def __init__(self, tensors: Iterable[str]):
        msg = f"Tensors not found: {tensors}. Use `tf.identity(t, name)` to name nodes in your graph."
        super().__init__(msg)


@dataclass
class OptimizeSavedModel(base.Job):
    """Converts SavedModel into an optimized protobuf for inference

    This job reads the input SavedModel, rename some nodes using the
    ``new_names`` argument (raises an error if some renames cannot be
    found), create placeholders given by ``feeds`` (and removes all
    other placeholders not in this list), and finally freezes the sub
    graph that produces the output tensor ``fetch``.

    When creating the original SavedModel, it is recommended to use
    ``tf.identity`` operators to mark some tensors as future feeds or
    fetches.

    WARNING: successful completion of this job is no guarantee that the
    exported Graph is correct. It is recommended to test the export in
    a separate job.

    Attributes
    ----------
    path_saved_model : str
        Path to directory containing SavedModel exports to convert
    path_optimized_model : str
        Path to directory that will contain the export
    graph_name : str
        Name of the output graph (name of the protobuf file)
    new_names : Dict[str, str]
        Mapping old names (SavedModel nodes) -> new names (export)
    blacklisted_variables : Union[str, List[str]]
        List of variable names not to include in the export
    feeds : Union[str, List[str]]
        List of nodes to use as inputs, or comma separated string.
    fetch : Union[str, List[str]]
        List of nodes to use as output, or comma separated string.
    """

    path_saved_model: str
    path_optimized_model: str
    graph_name: str
    feeds: Union[str, List[str]]
    fetch: Union[str, List[str]]
    new_names: Dict[str, str] = field(default_factory=dict)
    blacklisted_variables: List[str] = field(default_factory=list)

    def run(self):
        # Normalize feeds and fetch
        fetch = self.fetch.split(",") if isinstance(self.fetch, str) else self.fetch
        feeds = self.feeds.split(",") if isinstance(self.feeds, str) else self.feeds

        # Find latest SavedModel export in path_saved_model
        subdirs = [
            str(path) for path in Path(self.path_saved_model).iterdir() if path.is_dir() and "temp" not in str(path)
        ]
        latest = str(sorted(subdirs)[-1])
        LOGGER.info(f"Using SavedModel {latest}")

        # Reload SavedModel Graph, optimize and export
        with tf.Session(graph=tf.Graph()) as sess:
            meta_graph_def = tf.saved_model.loader.load(sess, ["serve"], latest)
            graph_def = meta_graph_def.graph_def

            # Add table initializer if present, or create it
            if INIT_ALL_TABLES in {node.name for node in graph_def.node}:
                fetch.append(INIT_ALL_TABLES)
            else:
                table_initializers = tf.get_collection(tf.GraphKeys.TABLE_INITIALIZERS)
                if table_initializers:
                    LOGGER.info(f"Adding {INIT_ALL_TABLES} Node to the graph")
                    table_init_op = tf.group(*table_initializers, name=INIT_ALL_TABLES)
                    node_def = table_init_op.node_def
                    graph_def.node.append(node_def)
                    fetch.append(INIT_ALL_TABLES)

            # Rename nodes
            graph_def = rename_nodes(graph_def, self.new_names)

            # Setup (create / remove) placeholders
            graph_def = make_placeholders(graph_def, feeds)

            # Keep only part of the graph that produces tensor 'fetch'
            graph_def = extract_sub_graph(graph_def, fetch)

            # Replace variables by constants
            graph_def = freeze_graph_with_def_protos(
                input_graph_def=graph_def,
                input_saver_def=None,
                input_checkpoint=None,
                output_node_names=",".join(fetch),
                restore_op_name=None,
                filename_tensor_name=None,
                output_graph=None,
                clear_devices=True,
                initializer_nodes=None,
                variable_names_blacklist=",".join(self.blacklisted_variables),
                input_saved_model_dir=latest,
                saved_model_tags=["serve"],
            )
            tf.io.write_graph(graph_def, logdir=self.path_optimized_model, name=self.graph_name, as_text=False)
            LOGGER.info(f"Optimized Model successfully exported to {self.path_optimized_model}/{self.graph_name}")


def rename_nodes(graph_def: tf.GraphDef, new_names: Dict[str, str]) -> tf.GraphDef:
    """Rename items in the graph to new ones defined in new_names

    Parameters
    ----------
    graph_def : tf.GraphDef
        Graph Definition
    new_names : Dict[str, str]
        Mapping old name -> new name

    Returns
    -------
    tf.GraphDef
        A copy of the input GraphDef with renamed nodes

    Raises
    ------
    TensorsNotFoundError
        If new_names refers to an node not found in graph_def
    """
    # Create copy of each node with a new name
    nodes = []
    for node in graph_def.node:
        new_node = tf.NodeDef()
        new_node.CopyFrom(node)
        nodes.append(new_node)
        match = re.match(r"^(?:cond(?:_\d+)?/)?(.+?)(?:_\d+)?$", node.name)
        if match and match.groups()[0] in new_names:
            new_name = new_names[match.groups()[0]]
            new_node.name = new_name
            LOGGER.info(f"Node renamed: {node.name} -> {new_node.name}")

    # Check that all new names were used
    if not set(new_names.values()) <= set(node.name for node in nodes):
        missing = set(new_names.values()) - set(node.name for node in nodes)
        raise TensorsNotFoundError(missing)

    # Update node references (inputs and location) to renamed nodes
    for node in nodes:
        for idx, name in enumerate(node.input):
            node.input[idx] = new_names[name] if name in new_names else name
        if "_class" in node.attr:
            attr = node.attr["_class"]
            for idx, item in enumerate(attr.list.s):
                loc_match = re.match(r"^loc:@(.+)$", item.decode())
                if loc_match and loc_match.groups()[0] in new_names:
                    new_name = new_names[loc_match.groups()[0]]
                    attr.list.s[idx] = f"loc:@{new_name}".encode()

    # Create Graph with renamed nodes
    new_graph = tf.GraphDef()
    new_graph.node.extend(nodes)
    return new_graph


def make_placeholders(graph_def: tf.GraphDef, names: List[str]) -> tf.GraphDef:
    """Create placeholders for names and remove other placeholders

    Parameters
    ----------
    graph_def : tf.GraphDef
        Graph definition
    names : List[str]
        Names of placeholders to keep / create for this graph

    Returns
    -------
    tf.GraphDef
        A copy of the input GraphDef with new placeholders

    Raises
    ------
    TensorsNotFoundError
        If names refers to a node that is not present
    """
    # Create copy of each node and change to Placeholder if in names
    nodes = []
    for node in graph_def.node:
        if node.name not in names and node.op == "Placeholder":
            LOGGER.info(f"Removing placeholder {node.name}")
            continue
        new_node = tf.NodeDef()
        if node.name in names and node.op != "Placeholder":
            LOGGER.info(f"Creating placeholder {node.name}")
            new_node.name = node.name
            new_node.op = "Placeholder"
            new_node.attr["shape"].CopyFrom(tf.AttrValue(shape=node.attr["_output_shapes"].list.shape[0]))
            new_node.attr["dtype"].CopyFrom(node.attr["T"])
        else:
            new_node.CopyFrom(node)
        nodes.append(new_node)

    # Check that all expected placeholders have been found
    if not set(names) <= set(node.name for node in nodes):
        missing = set(names) - set(node.name for node in nodes)
        raise TensorsNotFoundError(missing)

    # Create Graph with renamed nodes
    new_graph = tf.GraphDef()
    new_graph.node.extend(nodes)
    return new_graph
