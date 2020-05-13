"""Field"""

from collections import namedtuple
from typing import Any, Tuple

import numpy as np
import tensorflow as tf


class Field:
    """Convenient way to define fields for features.

    Attributes
    ----------
    default : Any
        Default value of the field for padding
    dtype : tf.DType
        Tensorflow type of the field (automatically inferred if string)
    name : str
        Name of the fields
    shape : Tuple
        Shape of the field
    """

    def __init__(self, name: str, shape: Tuple, dtype, default: Any = None):
        self.name = name
        self.shape = shape
        self.dtype = TensorType(dtype).tf
        self.default = default if default is not None else TensorType(dtype).default

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name}, {self.shape}, {self.dtype}, {self.default})"

    def __str__(self):
        return self.name

    def startswith(self, prefix: str):
        return self.name.startswith(prefix)

    def has_fixed_len(self) -> bool:
        return all(dim is not None for dim in self.shape)

    def has_var_len(self) -> bool:
        return any(dim is None for dim in self.shape)

    def as_placeholder(self, batch: bool = False) -> tf.placeholder:
        shape = tuple([None] + list(self.shape)) if batch else self.shape
        return tf.placeholder(dtype=self.dtype, shape=shape, name=self.name)

    def as_feature(self):
        if self.has_fixed_len():
            return tf.FixedLenFeature(shape=self.shape, dtype=self.dtype, default_value=self.default)
        else:
            return tf.VarLenFeature(dtype=self.dtype)


_TensorType = namedtuple("TensorType", ("tf, np, py, default, string"))


_TENSOR_TYPES = [
    _TensorType(tf.int32, np.int32, int, -1, "int32"),
    _TensorType(tf.int64, np.int64, int, -1, "int64"),
    _TensorType(tf.float32, np.float32, float, 0.0, "float32"),
    _TensorType(tf.float64, np.float64, float, 0.0, "float64"),
    _TensorType(tf.bool, np.bool, bool, True, "bool"),
    _TensorType(tf.string, np.dtype("S"), bytes, b"", "bytes"),
    _TensorType(tf.string, np.dtype("S"), str, b"", "string"),
]


def TensorType(dtype):
    """Return TensorType from Python, TensorFlow or NumPy type"""
    for tt in _TENSOR_TYPES:
        if dtype is tt.tf:
            return tt
        elif dtype is tt.np:
            return tt
        elif dtype is tt.py:
            return tt
        elif dtype == tt.string:
            return tt
    raise ValueError(f"TensorType not found `{dtype}`")
