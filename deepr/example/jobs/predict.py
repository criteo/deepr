"""Compute prediction on a dataset and log result."""

import logging
from typing import List, Callable
from dataclasses import dataclass

import tensorflow as tf
from tensorflow.python.platform import gfile

import deepr as dpr


LOGGER = logging.getLogger(__name__)


@dataclass
class Predict(dpr.jobs.Job):
    """Compute prediction on a dataset and log result."""

    path_model_pb: str
    feeds: List[str]
    fetches: List[str]
    input_fn: Callable[[], tf.data.Dataset]
    prepro_fn: Callable[[tf.data.Dataset, str], tf.data.Dataset]

    def run(self):
        with tf.Session(graph=tf.Graph()) as sess:
            # Import Graph Definition into the current graph
            with gfile.FastGFile(f"{self.path_model_pb}", "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                tf.import_graph_def(graph_def, name="model")

            # Retrieve feeds from Graph
            feeds = {}
            for feed in self.feeds:
                op_or_tensor = sess.graph.as_graph_element(f"model/{feed}")
                if isinstance(op_or_tensor, tf.Tensor):
                    tensor = op_or_tensor
                else:
                    if len(op_or_tensor.outputs) > 1:
                        raise ValueError(f"Found more than one tensor for operation {op_or_tensor}")
                    tensor = op_or_tensor.outputs[0]
                if not sess.graph.is_feedable(tensor):
                    raise ValueError(f"{feed} should be feedable but is not")
                feeds[feed] = tensor

            # Retrieve fetches from Graph
            fetches = {}
            for fetch in self.fetches:
                op_or_tensor = sess.graph.as_graph_element(f"model/{fetch}")
                if isinstance(op_or_tensor, tf.Tensor):
                    tensor = op_or_tensor
                else:
                    if len(op_or_tensor.outputs) > 1:
                        raise ValueError(f"Found more than one tensor for operation {op_or_tensor}")
                    tensor = op_or_tensor.outputs[0]
                if not sess.graph.is_feedable(tensor):
                    raise ValueError(f"{fetch} should be fetchable but is not")
                fetches[fetch] = tensor

            # Compute prediction on dataset
            dataset = self.prepro_fn(self.input_fn(), tf.estimator.ModeKeys.PREDICT)
            iterator = dataset.make_initializable_iterator()
            next_element = iterator.get_next()
            sess.run(tf.tables_initializer())
            sess.run(iterator.initializer)
            try:
                while True:
                    batch = sess.run(next_element)
                    preds = sess.run(fetches, {feeds[key]: tensor for key, tensor in batch.items()})
                    LOGGER.info({**batch, **preds})
            except tf.errors.OutOfRangeError:
                LOGGER.info(f"Reached end of {self.input_fn}")
                pass
