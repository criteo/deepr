"""Mean Metrics"""

from typing import Dict, List, Tuple
import logging

import tensorflow as tf

from deepr.metrics import base


LOGGER = logging.getLogger(__name__)


class Mean(base.Metric):
    """Finite Mean Metric"""

    def __init__(self, tensors: List[str] = None, pattern: str = None):
        self.tensors = tensors
        self.pattern = pattern

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tensors={self.tensors}, pattern={self.pattern})"

    def __call__(self, tensors: Dict[str, tf.Tensor]) -> Dict[str, Tuple]:
        tensors = base.get_scalars(tensors, names=self.tensors, pattern=self.pattern)
        return {name: tf.compat.v1.metrics.mean(value) for name, value in tensors.items()}


class FiniteMean(base.Metric):
    """Finite Mean Metric"""

    def __init__(self, tensors: List[str] = None, pattern: str = None):
        self.tensors = tensors
        self.pattern = pattern

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(tensors={self.tensors}, pattern={self.pattern})"

    def __call__(self, tensors: Dict[str, tf.Tensor]) -> Dict[str, Tuple]:
        tensors = base.get_scalars(tensors, names=self.tensors, pattern=self.pattern)
        return {name: finite_mean_metric(value, name) for name, value in tensors.items()}


def finite_mean_metric(value, name):
    """Compute Mean Metric"""
    # Variables
    acc = base.get_metric_variable(name=f"{name}_acc", shape=(), dtype=tf.float32)
    num = base.get_metric_variable(name=f"{name}_num", shape=(), dtype=tf.int64)

    # New Variables Values
    is_finite = tf.math.is_finite(value)
    new_acc = tf.cond(pred=is_finite, true_fn=lambda: acc + value, false_fn=lambda: acc)
    new_num = tf.cond(pred=is_finite, true_fn=lambda: num + 1, false_fn=lambda: num)

    # Return value and update op
    update_op = tf.group(tf.compat.v1.assign(acc, new_acc), tf.compat.v1.assign(num, new_num))
    val = tf.math.divide_no_nan(acc, tf.cast(num, dtype=tf.float32))
    return (val, update_op)


class DecayMean(base.Metric):
    """Decay Mean Metric"""

    def __init__(self, decay: float = 0.99, tensors: List[str] = None, pattern: str = None):
        self.decay = decay
        self.tensors = tensors
        self.pattern = pattern

        if decay > 1 or decay < 0:
            raise ValueError(f"decay must be between 0 and 1, but got {decay}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(decay={self.decay}, tensors={self.tensors}, pattern={self.pattern})"

    def __call__(self, tensors: Dict[str, tf.Tensor]) -> Dict[str, Tuple]:
        tensors = base.get_scalars(tensors, names=self.tensors, pattern=self.pattern)
        return {name: decay_mean_metric(value, self.decay, name) for name, value in tensors.items()}


def decay_mean_metric(value, decay: float, name: str):
    last = base.get_metric_variable(name=f"{name}_decayed_mean", shape=(), dtype=value.dtype)
    new_value = tf.cond(pred=tf.equal(last, 0), true_fn=lambda: value, false_fn=lambda: decay * last + (1.0 - decay) * value)
    update_op = tf.compat.v1.assign(last, new_value)
    return (last, update_op)
