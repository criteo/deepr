"""Abstract base class for reader"""

from abc import ABC, abstractmethod
import logging

import tensorflow as tf


LOGGER = logging.getLogger(__name__)


class Reader(ABC):
    """Interface for readers, similar to tensorflow_datasets"""

    def __call__(self) -> tf.data.Dataset:
        """Alias for as_dataset"""
        return self.as_dataset()

    def __iter__(self):
        dataset = self.as_dataset()
        iterator = dataset.make_initializable_iterator()
        next_element = iterator.get_next()
        with tf.Session() as sess:
            sess.run(tf.tables_initializer())
            sess.run(iterator.initializer)
            try:
                while True:
                    yield sess.run(next_element)
            except tf.errors.OutOfRangeError:
                msg = f"Reached end of {self}"
                LOGGER.info(msg)
                pass

    @abstractmethod
    def as_dataset(self) -> tf.data.Dataset:
        """Build a tf.data.Dataset"""
        raise NotImplementedError()


class DatasetReader(Reader):
    """Dummy dataset reader initialized with a tf.data.Dataset"""

    def __init__(self, dataset: tf.data.Dataset):
        super().__init__()
        self._dataset = dataset

    def __repr__(self):
        return f"DatasetReader(dataset={self._dataset})"

    def as_dataset(self) -> tf.data.Dataset:
        """Build a tf.data.Dataset"""
        return self._dataset


def from_dataset(dataset: tf.data.Dataset) -> DatasetReader:
    return DatasetReader(dataset)
