"""Compute prediction on a dataset and log result."""

import logging
from typing import List, Callable, Union
from dataclasses import dataclass

import tensorflow as tf

import deepr as dpr


LOGGER = logging.getLogger(__name__)


@dataclass
class Predict(dpr.jobs.Job):
    """Compute prediction on a dataset and log result."""

    path_model: str
    graph_name: str
    feeds: Union[str, List[str]]
    fetch: Union[str, List[str]]
    input_fn: Callable[[], tf.data.Dataset]
    prepro_fn: Callable[[tf.data.Dataset, str], tf.data.Dataset]

    def run(self):
        # Normalize feeds and fetch
        fetch = self.fetch.split(",") if isinstance(self.fetch, str) else self.fetch
        feeds = self.feeds.split(",") if isinstance(self.feeds, str) else self.feeds

        with tf.Session(graph=tf.Graph()) as sess:
            # Import Graph, retrieve feed and fetch tensors
            dpr.utils.import_graph_def(f"{self.path_model}/{self.graph_name}")
            feedable_tensors = dpr.utils.get_feedable_tensors(sess.graph, feeds)
            fetchable_tensors = dpr.utils.get_feedable_tensors(sess.graph, fetch)

            # Compute prediction on dataset
            dataset = self.prepro_fn(self.input_fn(), tf.estimator.ModeKeys.PREDICT)
            iterator = dataset.make_initializable_iterator()
            next_element = iterator.get_next()
            sess.run(tf.tables_initializer())
            sess.run(iterator.initializer)
            try:
                while True:
                    batch = sess.run(next_element)
                    preds = sess.run(
                        fetchable_tensors, {feedable_tensors[key]: tensor for key, tensor in batch.items()}
                    )
                    LOGGER.info({**batch, **preds})
            except tf.errors.OutOfRangeError:
                LOGGER.info(f"Reached end of {self.input_fn}")
                pass
