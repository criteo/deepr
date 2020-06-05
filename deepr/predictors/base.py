"""Base Class for Predictor."""

import abc
from typing import Dict, Union, Callable
import logging

import numpy as np
import tensorflow as tf


LOGGER = logging.getLogger(__name__)


class Predictor(abc.ABC):
    """Base Class for Predictor.

    Attributes
    ----------
    feed_tensors : Dict[str, tf.Tensor]
        Input tensors
    fetch_tensors : Dict[str, tf.Tensor]
        Output tensors
    session : tf.Session
        Tensorflow session
    """

    def __init__(self, session: tf.Session, feed_tensors: Dict[str, tf.Tensor], fetch_tensors: Dict[str, tf.Tensor]):
        self.session = session
        self.feed_tensors = feed_tensors
        self.fetch_tensors = fetch_tensors

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.session.close()

    def __call__(self, inputs: Union[Dict[str, np.array], Callable[[], tf.data.Dataset]]):
        if isinstance(inputs, dict):
            if not set(self.feed_tensors) <= set(inputs):
                raise KeyError(f"Missing keys in inputs: {set(self.feed_tensors) - set(inputs)} (inputs = {inputs})")
            return self.session.run(
                self.fetch_tensors, feed_dict={tensor: inputs[name] for name, tensor in self.feed_tensors.items()}
            )
        elif callable(inputs):

            def _gen():
                with self.session.graph.as_default():
                    dataset = inputs()
                    iterator = dataset.make_initializable_iterator()
                    next_element = iterator.get_next()
                    self.session.run(tf.tables_initializer())
                    self.session.run(iterator.initializer)
                    try:
                        while True:
                            input_dict = self.session.run(next_element)
                            output_dict = self.__call__(input_dict)
                            yield {**input_dict, **output_dict}
                    except tf.errors.OutOfRangeError:
                        pass

            return _gen()
        else:
            raise TypeError(f"Expected type dict or tf.data.Dataset but got {type(inputs)}")
