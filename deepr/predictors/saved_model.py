"""SavedModel Predictor."""

import logging
from typing import List, Union

import tensorflow as tf

from deepr.io.path import Path
from deepr.predictors import base
from deepr.utils.graph import get_feedable_tensors, get_fetchable_tensors


LOGGER = logging.getLogger(__name__)


def get_latest_saved_model(saved_model_dir: str) -> str:
    """Get latest sub directory in saved_model_dir.

    Parameters
    ----------
    saved_model_dir : str
        Path to directory containing saved model exports.

    Returns
    -------
    str
    """
    subdirs = [str(path) for path in Path(saved_model_dir).iterdir() if path.is_dir() and "temp" not in str(path)]
    return str(sorted(subdirs)[-1])


class SavedModelPredictor(base.Predictor):
    """SavedModel Predictor.

    Attributes
    ----------
    path : str
        Path to SavedModel directory
    feeds : Union[str, List[str]], optional
        Name of feed tensors or operations (inputs)
    fetches : Union[str, List[str]], optional
        Name of fetch tensors or operations (outputs)
    """

    def __init__(self, path: str, feeds: Union[str, List[str]] = None, fetches: Union[List[str], str] = None):
        # Store and normalize attributes
        self.path = path
        self.feeds = feeds.split(",") if isinstance(feeds, str) else feeds
        self.fetches = fetches.split(",") if isinstance(fetches, str) else fetches

        # Create session and import graph
        session = tf.Session(graph=tf.Graph())
        with session.graph.as_default():
            metagraph_def = tf.saved_model.load(session, ("serve",), path)
            signature_def = metagraph_def.signature_def[
                tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY
            ]

        # Retrieve feed tensors
        if self.feeds is None:
            LOGGER.info("Retrieving feeds from default signature def")
            feed_tensors = {
                name: session.graph.get_tensor_by_name(tensor_info.name)
                for name, tensor_info in signature_def.inputs.items()
            }
        else:
            feed_tensors = get_feedable_tensors(session.graph, self.feeds)

        # Retrieve fetch tensors
        if self.fetches is None:
            LOGGER.info("Retrieving fetches from default signature def")
            fetch_tensors = {
                name: session.graph.get_tensor_by_name(tensor_info.name)
                for name, tensor_info in signature_def.outputs.items()
            }
        else:
            fetch_tensors = get_fetchable_tensors(session.graph, self.fetches)

        super().__init__(session=session, feed_tensors=feed_tensors, fetch_tensors=fetch_tensors)
