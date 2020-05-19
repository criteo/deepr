"""Last value metric"""

from typing import Dict, List, Tuple

import tensorflow as tf

from deepr.metrics import base


class LastValue(base.Metric):
    """Last value Metric"""

    def __init__(self, tensors: List[str] = None):
        self.tensors = tensors

    def __call__(self, tensors: Dict[str, tf.Tensor]) -> Dict[str, Tuple]:
        if self.tensors is None:
            tensors = {key: tensor for key, tensor in tensors.items() if len(tensor.shape) == 0}
        else:
            tensors = {name: tensors[name] for name in self.tensors}
        return {name: last_value_metric(value, name) for name, value in tensors.items()}


def last_value_metric(value, name):
    last_value = base.get_metric_variable(name=f"{name}_last", shape=(), dtype=value.dtype)
    update_op = tf.assign(last_value, value)
    return (last_value, update_op)


class MaxValue(base.Metric):
    """Max value Metric"""

    def __init__(self, tensors: List[str] = None):
        self.tensors = tensors

    def __call__(self, tensors: Dict[str, tf.Tensor]) -> Dict[str, Tuple]:
        if self.tensors is None:
            tensors = {key: tensor for key, tensor in tensors.items() if len(tensor.shape) == 0}
        else:
            tensors = {name: tensors[name] for name in self.tensors}
        return {name: max_value_metric(value, name) for name, value in tensors.items()}


def max_value_metric(value, name):
    max_value = base.get_metric_variable(name=f"{name}_max", shape=(), dtype=value.dtype)
    update_op = tf.assign(max_value, tf.maximum(value, max_value))
    return (max_value, update_op)
