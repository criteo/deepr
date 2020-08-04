"""Build a dummy dataset of random (x, 2*x) as a tfrecord file"""

import logging
from dataclasses import dataclass

import tensorflow as tf
import numpy as np

import deepr as dpr


LOGGER = logging.getLogger(__name__)


@dataclass
class Build(dpr.jobs.Job):
    """Build a dummy dataset of random (x, 2*x) as a tfrecord file"""

    path_dataset: str
    num_examples: int = 1000

    def run(self):
        def _generator_fn():
            for _ in range(self.num_examples):
                x = np.random.random()
                yield {"x": x, "y": 2 * x}

        def _dict_to_example(data):
            features = {"x": dpr.readers.float_feature([data["x"]]), "y": dpr.readers.float_feature([data["y"]])}
            example = tf.train.Example(features=tf.train.Features(feature=features))
            return example

        with tf.python_io.TFRecordWriter(self.path_dataset) as writer:
            for data in _generator_fn():
                example = _dict_to_example(data)
                writer.write(example.SerializeToString())

        LOGGER.info(f"Wrote dataset to '{self.path_dataset}'")
