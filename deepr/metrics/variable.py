"""Variable Value Metric."""

from typing import Dict, Tuple
import logging

import tensorflow as tf

from deepr.metrics import base


LOGGER = logging.getLogger(__name__)


class VariableValue(base.Metric):
    """Variable Value Metric.

    Return value of variable created with ``tf.get_variable`` if the
    variable is a scalar. Otherwise, return the norm of that variable.

    Attributes
    ----------
    name : str
        Variable name
    """

    def __init__(self, name: str):
        self.name = name

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"

    def __call__(self, tensors: Dict[str, tf.Tensor]) -> Dict[str, Tuple]:
        # pylint: disable=unused-argument
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            variable = tf.get_variable(self.name)
            if variable.shape == ():
                metric = variable
            else:
                LOGGER.info(f"Variable {self.name} has shape {variable.shape}, computing norm instead")
                metric = tf.global_norm([variable])
        return {self.name: (metric, tf.no_op())}
