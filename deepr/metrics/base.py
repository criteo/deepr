"""Base class for Metrics"""

from abc import ABC
from typing import Dict, Tuple

import tensorflow as tf


class Metric(ABC):
    """Base class for Metrics"""

    def __call__(self, tensors: Dict[str, tf.Tensor]) -> Dict[str, Tuple]:
        raise NotImplementedError()


def get_metric_variable(name: str, shape: Tuple, dtype):
    return tf.get_variable(
        name=name,
        shape=shape,
        dtype=dtype,
        initializer=tf.constant_initializer(value=0),
        collections=[tf.GraphKeys.LOCAL_VARIABLES, tf.GraphKeys.METRIC_VARIABLES],
        trainable=False,
    )
