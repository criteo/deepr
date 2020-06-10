"""Base class for Metrics"""

from abc import ABC
from typing import Dict, Tuple, List
import re
import logging

import tensorflow as tf


LOGGER = logging.getLogger(__name__)


class Metric(ABC):
    """Base class for Metrics"""

    def __call__(self, tensors: Dict[str, tf.Tensor]) -> Dict[str, Tuple]:
        raise NotImplementedError()

    def __repr__(self) -> str:
        return self.__class__.__name__


def get_tensors(tensors: Dict[str, tf.Tensor], names: List[str] = None, pattern: str = None) -> Dict[str, tf.Tensor]:
    """Extract tensors with names / pattern from tensors dictionary

    Parameters
    ----------
    tensors : Dict[str, tf.Tensor]
        Dictionary
    names : List[str], optional
        Names in tensors
    pattern : str, optional
        Pattern for re.match

    Returns
    -------
    Dict[str, tf.Tensor]
    """
    found = dict()
    if names is not None:
        found.update({name: tensors[name] for name in names})
    if pattern is not None:
        found.update({name: tensor for name, tensor in tensors.items() if re.match(pattern, name)})
    return found


def keep_scalars(tensors: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Remove non-scalar tensors from tensors.

    Parameters
    ----------
    tensors : Dict[str, tf.Tensor]
        Dictionary
    """
    filtered = {}
    for name, tensor in tensors.items():
        if len(tensor.shape) != 0:
            LOGGER.warning(f"Remove {name}, shape={tensor.shape} (must be scalar).")
            continue
        filtered[name] = tensor
    return filtered


def get_scalars(tensors: Dict[str, tf.Tensor], names: List[str] = None, pattern: str = None) -> Dict[str, tf.Tensor]:
    """Retrieve scalars from tensors.

    Parameters
    ----------
    tensors : Dict[str, tf.Tensor]
        Dictionary
    names : List[str], optional
        Tensor names
    pattern : str, optional
        Pattern for re.match

    Returns
    -------
    Dict[str, tf.Tensor]
    """
    if names is not None or pattern is not None:
        tensors = get_tensors(tensors=tensors, names=names, pattern=pattern)
    else:
        tensors = {key: tensor for key, tensor in tensors.items() if len(tensor.shape) == 0}
    return keep_scalars(tensors)


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


def get_metric_variable(name: str, shape: Tuple, dtype) -> tf.Variable:
    return tf.get_variable(
        name=sanitize_metric_name(name),
        shape=shape,
        dtype=dtype,
        initializer=tf.constant_initializer(value=0),
        collections=[tf.GraphKeys.LOCAL_VARIABLES, tf.GraphKeys.METRIC_VARIABLES],
        trainable=False,
    )
