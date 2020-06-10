"""Last value metric"""

import logging
from typing import Dict, List, Tuple

import tensorflow as tf

from deepr.metrics import base


LOGGER = logging.getLogger(__name__)


class LastValue(base.Metric):
    """Last value Metric"""

    def __init__(self, tensors: List[str] = None, pattern: str = None):
        self.tensors = tensors
        self.pattern = pattern

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tensors={self.tensors}, pattern={self.pattern})"

    def __call__(self, tensors: Dict[str, tf.Tensor]) -> Dict[str, Tuple]:
        tensors = base.get_scalars(tensors, names=self.tensors, pattern=self.pattern)
        LOGGER.info(f"{self} -> {', '.join(tensors.keys())}")
        return {name: (tensor, tf.no_op()) for name, tensor in tensors.items()}


class MaxValue(base.Metric):
    """Max value Metric"""

    def __init__(self, tensors: List[str] = None, pattern: str = None):
        self.tensors = tensors
        self.pattern = pattern

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tensors={self.tensors}, pattern={self.pattern})"

    def __call__(self, tensors: Dict[str, tf.Tensor]) -> Dict[str, Tuple]:
        tensors = base.get_scalars(tensors, names=self.tensors, pattern=self.pattern)
        LOGGER.info(f"{self} -> {', '.join(tensors.keys())}")
        return {name: max_value_metric(value, name) for name, value in tensors.items()}


def max_value_metric(value, name):
    max_value = base.get_metric_variable(name=f"{name}_max", shape=(), dtype=value.dtype)
    update_op = tf.assign(max_value, tf.maximum(value, max_value))
    return (max_value, update_op)
