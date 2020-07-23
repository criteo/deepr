"""Core Layers"""

from typing import Tuple, List, Union

import tensorflow as tf

from deepr.layers import base
from deepr.utils.broadcasting import make_same_shape


class Sum(base.Layer):
    """Sum Layer"""

    def __init__(self, n_in: int = 2, **kwargs):
        super().__init__(n_in=n_in, n_out=1, **kwargs)

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer"""
        tensors = make_same_shape(tensors, broadcast=False)
        acc = 0
        for inp in tensors:
            acc += inp
        return acc


class Product(base.Layer):
    """Product Layer"""

    def __init__(self, n_in: int = 2, **kwargs):
        super().__init__(n_in=n_in, n_out=1, **kwargs)

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer"""
        tensors = make_same_shape(tensors, broadcast=False)
        acc = 1
        for inp in tensors:
            acc *= inp
        return acc


class DotProduct(base.Layer):
    """Dot Product on the last dimension of the input vectors.

    It will add missing dimensions to the before last dimension. For
    example, if

        - t1: shape = [batch, num_target, 100]
        - t2: shape = [batch, 100]

    It will return

        - t: shape = [batch, num_target], where
            t[i, j] = sum_k(t1[i, k] * t2[i, j, k])
    """

    def __init__(self, n_in: int = 2, **kwargs):
        super().__init__(n_in=n_in, n_out=1, **kwargs)

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer"""
        # Add missing dimensions to tensors to make them compatible
        t1, t2 = tensors
        if len(t1.shape) < len(t2.shape):
            t1, t2 = t2, t1
        ndims = len(t1.shape) - len(t2.shape)
        for _ in range(ndims):
            t2 = tf.expand_dims(t2, axis=-2)

        # Matmul can be used for dot product if at least one dummy dim
        if ndims:
            return tf.squeeze(tf.matmul(t1, t2, transpose_b=True), axis=[-1])
        else:
            t1 = tf.expand_dims(t1, axis=-2)
            t2 = tf.expand_dims(t2, axis=-1)
            return tf.squeeze(tf.matmul(t1, t2), axis=[-2, -1])


class Dense(base.Layer):
    """Dense Layer"""

    def __init__(
        self,
        units: int,
        inputs: Union[str, Tuple[str, ...], List[str]] = None,
        outputs: Union[str, Tuple[str, ...], List[str]] = None,
        name: str = None,
        **kwargs
    ):
        super().__init__(n_in=1, n_out=1, inputs=inputs, outputs=outputs, name=name)
        self.units = units
        self._kwargs = kwargs

    def forward(self, tensors, mode: str = None):
        return tf.layers.dense(tensors, units=self.units, **self._kwargs)


@base.layer(n_in=2, n_out=1)
def Add(tensors):
    """Add two tensors of any compatible shapes."""
    t1, t2 = make_same_shape(tensors, broadcast=False)
    return t1 + t2


@base.layer(n_in=2, n_out=1)
def Concat(tensors, axis: int = -1):
    """Concatenate tensors on axis"""
    return tf.concat(tensors, axis=axis)


@base.layer(n_in=2, n_out=1)
def LogicalAnd(tensors):
    """Perform logical_and on two tensors of compatible shapes."""
    t1, t2 = make_same_shape(tensors, broadcast=False)
    return tf.logical_and(t1, t2)


@base.layer(n_in=2, n_out=1)
def LogicalOr(tensors):
    """Perform logical_or on two tensors of compatible shapes."""
    t1, t2 = make_same_shape(tensors, broadcast=False)
    return tf.logical_or(t1, t2)


@base.layer(n_in=1, n_out=1)
def ToFloat(tensors):
    """Cast tensor to float32"""
    return tf.cast(tensors, tf.float32)


@base.layer(n_in=1, n_out=1)
def ExpandDims(tensors, axis: int = -1):
    return tf.expand_dims(tensors, axis=axis)


@base.layer(n_in=1, n_out=1)
def Scale(tensors: tf.Tensor, multiplier: float):
    """Scale tensor by multiplier."""
    return tf.multiply(tensors, multiplier)


class Identity(base.Layer):
    """Identity Layer"""

    def __init__(self, inputs: Union[str, Tuple[str, ...], List[str]] = None, name: str = None):
        super().__init__(n_in=1, n_out=1, inputs=inputs, outputs=inputs, name=name)

    def forward(self, tensors, mode: str = None):
        return tf.identity(tensors, name=self.name)


class Conv1d(base.Layer):
    """Conv1d Layer"""

    def __init__(
        self,
        filters: int,
        kernel_size: int,
        use_bias: bool = True,
        activation=None,
        inputs: Union[str, Tuple[str, ...], List[str]] = None,
        outputs: Union[str, Tuple[str, ...], List[str]] = None,
        name: str = None,
        **kwargs
    ):
        super().__init__(n_in=1, n_out=1, inputs=inputs, outputs=outputs, name=name)
        self.filters = filters
        self.kernel_size = kernel_size
        self.use_bias = use_bias
        self.activation = activation
        self._kwargs = kwargs

    def forward(self, tensors, mode: str = None):
        return tf.layers.conv1d(
            inputs=tensors,
            filters=self.filters,
            kernel_size=self.kernel_size,
            activation=self.activation,
            use_bias=self.use_bias,
            **self._kwargs
        )


class Softmax(base.Layer):
    """Apply softmax to the last dimension of tensor with filtering masked values"""

    def __init__(self, n_in: int = 2, n_out=1, **kwargs):
        super().__init__(n_in=n_in, n_out=n_out, **kwargs)

    def forward(self, tensors, mode: str = None):
        """Forward method of the layer"""
        tensor, mask = tensors
        mask = tf.cast(mask, tf.float32)
        tensor_exp = tf.exp(tensor - tf.reduce_max(tensor * mask, axis=-1, keepdims=True))
        sum_tensor_exp = tf.reduce_sum(tf.multiply(tensor_exp, mask), axis=-1, keepdims=True)
        return tf.div_no_nan(tensor_exp, sum_tensor_exp) * mask
