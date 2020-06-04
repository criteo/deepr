"""Proto Predictor."""

import logging
from typing import List, Union

import tensorflow as tf

from deepr.predictors import base
from deepr.utils.graph import import_graph_def, get_feedable_tensors, get_fetchable_tensors


LOGGER = logging.getLogger(__name__)


class ProtoPredictor(base.Predictor):
    """Proto Predictor.

    Attributes
    ----------
    path : str
        Path to .pb file
    feeds : Union[str, List[str]]
        Name of feed tensors or operations (inputs)
    fetches : Union[str, List[str]]
        Name of fetch tensors or operations (outputs)
    """

    def __init__(self, path: str, feeds: Union[str, List[str]], fetches: Union[List[str], str]):
        # Store and normalize attributes
        self.path = path
        self.feeds = feeds.split(",") if isinstance(feeds, str) else feeds
        self.fetches = fetches.split(",") if isinstance(fetches, str) else fetches

        # Create session and import graph under scope "model"
        session = tf.Session(graph=tf.Graph())
        with session.graph.as_default():
            import_graph_def(path, name="model")

        # Retrieve feed tensors (add and remove "model" scope prefix)
        feed_tensors = {
            "/".join(name.split("/")[1:]): tensor
            for name, tensor in get_feedable_tensors(session.graph, [f"model/{name}" for name in self.feeds]).items()
        }

        # Retrieve fetch tensors (add and remove "model" scope prefix)
        fetch_tensors = {
            "/".join(name.split("/")[1:]): tensor
            for name, tensor in get_fetchable_tensors(session.graph, [f"model/{name}" for name in self.fetches]).items()
        }
        super().__init__(session=session, feed_tensors=feed_tensors, fetch_tensors=fetch_tensors)
