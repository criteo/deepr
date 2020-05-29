"""Base class for Metrics"""

from abc import ABC
from typing import Dict, Tuple

import tensorflow as tf


class Metric(ABC):
    """Base class for Metrics"""

    def __call__(self, tensors: Dict[str, tf.Tensor]) -> Dict[str, Tuple]:
        raise NotImplementedError()


def sanitize_metric_name(name: str) -> str:
    """Sanitize scope/variable name for tensorflow and mlflow

    This is needed as sometimes variables automatically created while
    building layers contain forbidden characters
    >>> from tensorflow.python.framework.ops import _VALID_SCOPE_NAME_REGEX as TF_VALID_REGEX
    >>> from mlflow.utils.validation import _VALID_PARAM_AND_METRIC_NAMES as MLFLOW_VALID_REGEX
    >>> from deepr.metrics import sanitize_metric_name
    >>> kernel_variable_name = 'my_layer/kernel:0'
    >>> bool(TF_VALID_REGEX.match(kernel_variable_name))
    False
    >>> bool(MLFLOW_VALID_REGEX.match(kernel_variable_name))
    False
    >>> bool(TF_VALID_REGEX.match(sanitize_metric_name(kernel_variable_name)))
    True
    >>> bool(MLFLOW_VALID_REGEX.match(sanitize_metric_name(kernel_variable_name)))
    True
    """
    name = name.replace(":", "-")
    return name


def get_metric_variable(name: str, shape: Tuple, dtype):
    return tf.get_variable(
        name=sanitize_metric_name(name),
        shape=shape,
        dtype=dtype,
        initializer=tf.constant_initializer(value=0),
        collections=[tf.GraphKeys.LOCAL_VARIABLES, tf.GraphKeys.METRIC_VARIABLES],
        trainable=False,
    )
