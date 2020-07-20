"""Masking Layers"""

from enum import Enum
from typing import Any, Tuple

import tensorflow as tf

from deepr.layers import base


class BooleanReduceMode(Enum):
    """Boolean Reduce Mode"""

    OR = "or"
    AND = "and"


class Equal(base.Layer):
    """Equal Layer"""

    def __init__(self, values: Tuple[Any, ...], reduce_mode: BooleanReduceMode = BooleanReduceMode.OR, **kwargs):
        super().__init__(n_in=1, n_out=1, **kwargs)
        self.values = values
        self.reduce_mode = reduce_mode

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer"""
        mask = None
        for value in self.values:
            value_mask = tf.equal(tensors, value)
            if mask is None:
                mask = value_mask
            elif self.reduce_mode == BooleanReduceMode.OR:
                mask = tf.logical_or(mask, value_mask)
            elif self.reduce_mode == BooleanReduceMode.AND:
                mask = tf.logical_and(mask, value_mask)
            else:
                msg = f"{self.reduce_mode} not recognized"
                raise ValueError(msg)

        if mask is None:
            msg = "mask should not be None."
            raise ValueError(msg)

        return mask


class NotEqual(base.Layer):
    """Not Equal Layer"""

    def __init__(self, values: Tuple[Any, ...], reduce_mode: BooleanReduceMode = BooleanReduceMode.AND, **kwargs):
        super().__init__(n_in=1, n_out=1, **kwargs)
        self.values = values
        self.reduce_mode = reduce_mode

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer"""
        mask = None
        for value in self.values:
            value_mask = tf.not_equal(tensors, value)
            if mask is None:
                mask = value_mask
            elif self.reduce_mode == BooleanReduceMode.OR:
                mask = tf.logical_or(mask, value_mask)
            elif self.reduce_mode == BooleanReduceMode.AND:
                mask = tf.logical_and(mask, value_mask)
            else:
                msg = f"{self.reduce_mode} not recognized"
                raise ValueError(msg)

        if mask is None:
            msg = "mask should not be None."
            raise ValueError(msg)

        return mask


class BooleanMask(base.Layer):
    """Boolean Mask Layer"""

    def __init__(self, **kwargs):
        super().__init__(n_in=2, n_out=1, **kwargs)

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer"""
        tensor, mask = tensors
        return tf.boolean_mask(tensor, mask)
