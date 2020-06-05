"""Base class for writers."""

import abc
import logging

import tensorflow as tf


LOGGER = logging.getLogger(__name__)


class Writer(abc.ABC):
    """Base class for writers."""

    @abc.abstractmethod
    def write(self, dataset: tf.data.Dataset):
        raise NotImplementedError()
