"""Export xla compatible model metadata from a saved model"""

from dataclasses import dataclass
import logging
from typing import List, Dict

import tensorflow as tf

from deepr.io import Path
from deepr.jobs import base
from deepr.utils import import_graph_def
import deepr.utils.tf2xla_pb2 as xla

LOGGER = logging.getLogger(__name__)


@dataclass
class ExportXlaModelMetadata(base.Job):
    """Export xla compatible model metadata from a saved model

    Attributes
    ----------
    path_optimized_model : str
        Path to directory containing optimized saved model exports to convert
    path_metadata : str
        Path to directory that will contain the metadata
    graph_name : str
        Name of the saved model graph (name of the protobuf file)
    metadata_name : str
        Name of the metadata file
    feed_shapes : Dict[str, List[int]]
        Shapes of feeds to expose
    fetch_shapes : Dict[str, List[int]]
        Shapes of fetches to expose
    """

    path_optimized_model: str
    path_metadata: str
    graph_name: str
    metadata_name: str
    feed_shapes: Dict[str, List[int]]
    fetch_shapes: Dict[str, List[int]]

    def run(self):
        # Create session and import graph under scope "model"
        session = tf.Session(graph=tf.Graph())
        with session.graph.as_default():
            import_graph_def(f"{self.path_optimized_model}/{self.graph_name}", name="")

            feed_nodes = get_nodes(session.graph_def, self.feed_shapes.keys())
            fetch_nodes = get_nodes(session.graph_def, self.fetch_shapes.keys())

            meta = xla.Config()
            for name, node in feed_nodes.items():
                add_metadata_item(meta.feed.add(), node, self.feed_shapes[name])
            for name, node in fetch_nodes.items():
                add_metadata_item(meta.fetch.add(), node, self.fetch_shapes[name])

            with Path(f"{self.path_metadata}/{self.metadata_name}").open("wb") as file:
                file.write(str(meta).encode("ascii"))
            LOGGER.info(f"Metadata successfully saved to {self.path_metadata}/{self.metadata_name}")


def get_nodes(graph_def: tf.GraphDef, names):
    nodes = {}
    for node in graph_def.node:
        if node.name in names:
            nodes[node.name] = node
    missing = set(names) - set(nodes.keys())

    if len(missing) > 0:
        raise Exception(f"Could not find these nodes in the graph : {missing}")
    return nodes


def add_metadata_item(item, node, target_shape=None):
    """add_metadata_item"""

    # There are two names:
    #   * Node Name @ ID (mandatory)
    #   * Name
    item.id.node_name = node.name
    # ID is the latest name of the Node Name.
    item.name = node.name.split("/")[-1]

    if target_shape is not None:
        if len(node.attr["shape"].shape.dim) != len(target_shape):
            raise ValueError(
                f"Source shape {node.attr['shape'].shape.dim} and target shape "
                f"{target_shape} are not compatible (different length)"
            )
        item.shape.CopyFrom(tf.TensorShape(target_shape).as_proto())

    # Type is optional but is nice to have to avoid any ambiguities.
    # If type is not defined, then use FLOAT32.
    item.type = 0 if node.attr["dtype"] is None else node.attr["dtype"].type
