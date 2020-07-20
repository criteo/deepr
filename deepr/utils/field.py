"""Field."""

from collections import namedtuple
from typing import Any, Tuple, Union

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
        Name of the field
    sequence : bool
        If True, the field represents a sequence.

        Used for ``tf.Example`` message serialization : if ``sequence``
        is ``True``, the field with be stored in the ``feature_list``
        entry of a ``tf.train.SequenceExample``.

        Automatically set if not given : ``True`` if ``shape``'s first
        dimension is ``None``.

    shape : Tuple
        Shape of the field
    """

    def __init__(self, name: str, shape: Tuple, dtype, default: Any = None, sequence: bool = None):
        self.name = name
        self.shape = tuple(shape)
        self.dtype = TensorType(dtype).tf
        self.default = default if default is not None else TensorType(dtype).default
        self.sequence = (
            sequence if sequence is not None else (any(dim is None for dim in shape) if len(shape) == 2 else False)
        )

        if self.sequence and not self.shape:
            msg = f"sequence=True but shape={self.shape}: expected at least one dimension."
            raise ValueError(msg)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.name}, {self.shape}, {self.dtype}, {self.default})"

    def __str__(self):
        return self.name

    def __hash__(self):
        return hash(self.name)

    @property
    def feature_specs(self):
        """Return feature specs for parsing Example messages."""
        if not self.is_featurizable():
            raise ValueError(f"{self} is not featurizable, no feature specs.")
        if self.sequence:
            if any(dim is None for dim in self.shape[1:]):
                return tf.io.VarLenFeature(dtype=self.dtype)
            else:
                return tf.io.FixedLenSequenceFeature(shape=self.shape[1:], dtype=self.dtype)
        else:
            if any(dim is None for dim in self.shape):
                return tf.io.VarLenFeature(dtype=self.dtype)
            else:
                return tf.io.FixedLenFeature(shape=self.shape, dtype=self.dtype)

    def is_sparse(self) -> bool:
        if self.is_featurizable():
            return isinstance(self.feature_specs, tf.io.VarLenFeature)
        return False

    def is_featurizable(self) -> bool:
        if self.sequence:
            if len(self.shape) > 2 and any(dim is None for dim in self.shape[1:]):
                return False
        else:
            if len(self.shape) > 1 and any(dim is None for dim in self.shape):
                return False
        return True

    def startswith(self, prefix: str):
        return self.name.startswith(prefix)

    def as_placeholder(self, batch: bool = False) -> tf.placeholder:
        shape = tuple([None] + list(self.shape)) if batch else self.shape
        return tf.placeholder(dtype=self.dtype, shape=shape, name=self.name)

    def to_feature(self, value: np.array) -> Union[tf.train.Feature, tf.train.FeatureList]:
        """Convert value to tf.train.Feature or tf.train.FeatureList.

        For shapes with more than 2 dimensions, uses ``np.ravel`` to
        flatten tensors in a list of values. Note that because
        ``tf.Example`` uses row-major to parse list of values, we make
        sure to use the same order with NumPy.

        For that reason, if any of the dimensions is not set (i.e. is
        ``None``), a ``ValueError`` is raised.

        Parameters
        ----------
        value : np.array
            Tensor values

        Returns
        -------
        tf.train.FeatureList
            If ``sequence`` is ``True``
        tf.train.Feature
            If ``sequence`` is ``False``

        Raises
        ------
        ValueError
            If ``sequence``, ``len(shape) > 2`` and one of the non-first
            dimensions is not set (i.e. is ``None``).
            If not ``sequence``, ``len(shape) > 2`` and any of the
            dimensions is not set (i.e. is ``None``).
        """

        def _to_feature(val):
            """Return tf.train.Feature"""
            if self.dtype is tf.int32 or self.dtype is tf.int64:
                return tf.train.Feature(int64_list=tf.train.Int64List(value=val))
            if self.dtype is tf.float32 or self.dtype is tf.float64:
                return tf.train.Feature(float_list=tf.train.FloatList(value=val))
            if self.dtype is tf.string:
                return tf.train.Feature(bytes_list=tf.train.BytesList(value=val))
            else:
                raise TypeError()

        if self.sequence:
            if len(self.shape) == 0:
                msg = f"sequence=True but shape={self.shape}: expected at least one dimension."
                raise ValueError(msg)
            if len(self.shape) == 1:
                return tf.train.FeatureList(feature=[_to_feature([val]) for val in value])
            if len(self.shape) == 2:
                return tf.train.FeatureList(feature=[_to_feature(val) for val in value])
            if any(dim is None for dim in self.shape[1:]):
                msg = f"Unable to convert field {self} to feature. If ndim > 2, dimensions must be static."
                raise ValueError(msg)
            return tf.train.FeatureList(feature=[_to_feature(np.ravel(val, order="C")) for val in value])
        else:
            if len(self.shape) == 0:
                return _to_feature([value])
            if len(self.shape) == 1:
                return _to_feature(value)
            if any(dim is None for dim in self.shape):
                msg = f"Unable to convert field {self} to feature. If ndim > 2, dimensions must be static."
                raise ValueError(msg)
            return _to_feature(np.ravel(value, order="C"))


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
    # pylint: disable=invalid-name
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
