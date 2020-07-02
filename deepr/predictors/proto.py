"""Proto Predictor."""

import logging
from typing import List, Union

import tensorflow as tf

from deepr.predictors import base
from deepr.utils.graph import (
    import_graph_def,
    get_feedable_tensors,
    get_fetchable_tensors,
    get_by_name,
    INIT_ALL_TABLES,
)


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

        # Create session and import graph
        session = tf.Session(graph=tf.Graph())
        with session.graph.as_default():
            import_graph_def(path)

            # Run Table initializer if present in the graph
            init_all_tables = get_by_name(session.graph, INIT_ALL_TABLES)
            if init_all_tables:
                LOGGER.info(f"Running {INIT_ALL_TABLES}")
                session.run(init_all_tables)

        # Retrieve feeds and fetches
        feed_tensors = get_feedable_tensors(session.graph, self.feeds)
        fetch_tensors = get_fetchable_tensors(session.graph, self.fetches)

        super().__init__(session=session, feed_tensors=feed_tensors, fetch_tensors=fetch_tensors)
