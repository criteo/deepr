"""StepCounter Metric"""

from typing import Dict, Tuple

import tensorflow as tf

from deepr.metrics import base


class StepCounter(base.Metric):
    """StepCounter Metric"""

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"

    def __call__(self, tensors: Dict[str, tf.Tensor]) -> Dict[str, Tuple]:
        # pylint: disable=unused-argument
        value = base.get_metric_variable(name=self.name, shape=(), dtype=tf.int64)
        update_op = tf.assign(value, value + 1)
        return {self.name: (value, update_op)}
