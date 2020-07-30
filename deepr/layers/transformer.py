# pylint: disable=no-value-for-parameter,invalid-name,unexpected-keyword-arg
"""Transformer Model."""

import logging
from typing import Tuple

import numpy as np
import tensorflow as tf

from deepr.layers import base
from deepr.layers.combinators import Sequential, Select, Scope
from deepr.layers.dropout import SpatialDropout1D, Dropout
from deepr.layers.slice import SliceLastPadded
from deepr.layers.core import Conv1d, Dense, Add, Scale


LOGGER = logging.getLogger(__name__)


def Transformer(
    dim: int,
    num_heads: int = 4,
    encoding_blocks: int = 2,
    dim_head: int = 128,
    residual_connection: bool = True,
    use_layer_normalization: bool = True,
    event_dropout_rate: float = 0.0,
    use_feedforward: bool = True,
    ff_dropout_rate: float = 0.0,
    ff_normalization: bool = False,
    scale: bool = False,
    use_positional_encoding: bool = True,
    trainable_positional_encoding: bool = True,
    use_look_ahead_mask: bool = True,
    inputs: Tuple[str, str] = ("inputEmbeddings", "inputMask"),
    outputs: str = "userEmbeddings",
) -> base.Layer:
    """Transformer Model."""
    return Sequential(
        Select(n_in=2, inputs=inputs, outputs=("inputEmbeddings", "inputMask")),
        SpatialDropout1D(inputs="inputEmbeddings", outputs="inputEmbeddingsDropout", dropout_rate=event_dropout_rate),
        AttentionMask(inputs="inputMask", outputs="mask", use_look_ahead_mask=use_look_ahead_mask),
        (
            Scale(inputs="inputEmbeddingsDropout", outputs="inputEnc", multiplier=(num_heads * dim_head) ** 0.5)
            if scale
            else Select(inputs="inputEmbeddingsDropout", outputs="inputEnc")
        ),
        (
            PositionalEncoding(inputs="inputEnc", outputs="inputEnc", trainable=trainable_positional_encoding,)
            if use_positional_encoding
            else []
        ),
        [
            Scope(
                Sequential(
                    SelfMultiheadAttention(
                        inputs=("inputEnc", "mask"),
                        outputs="inputEnc",
                        dim_head=dim_head,
                        num_heads=num_heads,
                        residual_connection=residual_connection,
                    ),
                    (
                        Scope(Normalization(inputs="inputEnc", outputs="inputEnc"), "attention_norm")
                        if use_layer_normalization and not (not use_feedforward and block_id == encoding_blocks - 1)
                        else []
                    ),
                    (
                        FeedForward(
                            inputs="inputEnc",
                            outputs="inputEnc",
                            units_inner=(num_heads * dim_head),
                            units_readout=(num_heads * dim_head),
                            dim=dim,
                            dropout_rate=ff_dropout_rate,
                        )
                        if use_feedforward
                        else []
                    ),
                    (
                        Scope(Normalization(inputs="inputEnc", outputs="inputEnc"), "ff_norm")
                        if use_feedforward and ff_normalization and block_id != encoding_blocks - 1
                        else []
                    ),
                ),
                f"block_{block_id}",
            )
            for block_id in range(encoding_blocks)
        ],
        SliceLastPadded(inputs=("inputEnc", "inputMask"), outputs=outputs),
    )


def FeedForward(
    inputs: str, outputs: str, units_inner: int, units_readout: int, dim: int, dropout_rate: float,
):
    """FeedForward Layer."""
    if inputs == "_x":
        raise ValueError("Cannot use name '_x' for inputs (used as intermediary node).")

    return Sequential(
        Select(inputs=inputs, outputs="_x"),
        Dropout(inputs="_x", outputs="_x", dropout_rate=dropout_rate),
        Conv1d(inputs="_x", outputs="_x", filters=units_inner, kernel_size=1, activation=tf.nn.relu, use_bias=True),
        Dropout(inputs="_x", outputs="_x", dropout_rate=dropout_rate),
        Conv1d(inputs="_x", outputs="_x", filters=units_readout, kernel_size=1, activation=None, use_bias=True),
        Dropout(inputs="_x", outputs="_x", dropout_rate=dropout_rate),
        Dense(inputs="_x", outputs="_x", units=dim),
        Add(inputs=(inputs, "_x"), outputs=outputs),
    )


@base.layer(n_in=1, n_out=1)
def Normalization(tensors: tf.Tensor, epsilon=1e-8):
    """Normalization Layer."""
    params_shape = tensors.get_shape()[-1:]
    mean, variance = tf.nn.moments(tensors, [-1], keep_dims=True)
    beta = tf.get_variable("beta", shape=params_shape, initializer=tf.zeros_initializer())
    gamma = tf.get_variable("gamma", shape=params_shape, initializer=tf.ones_initializer())
    normalized = (tensors - mean) / ((variance + epsilon) ** 0.5)
    return gamma * normalized + beta


@base.layer(n_in=1, n_out=1)
def PositionalEncoding(tensors: tf.Tensor, max_sequence_length=10000, trainable=False):
    """Add Positional Embeddings.

    Parameters
    ----------
    tensors : tf.Tensor
        Input tensor, [batch_size, sequence_length, emb_dim]
    use_positional_encoding : bool
        Use this layer in case of True, skip in case of False
    max_sequence_length : int
        Expected that input tensor length doesn't exceed the
        `max_sequence_length` limit
    trainable : bool
        Train / not train position encoding
    """
    with tf.variable_scope("positional_encoding"):
        emb_dim = tensors.get_shape().as_list()[-1]

        if trainable:
            initializer = None
        else:
            position_embeddings_np = np.array(
                [
                    [pos / np.power(10000, (i - i % 2) / emb_dim) for i in range(emb_dim)]
                    for pos in range(max_sequence_length)
                ]
            )
            position_embeddings_np[:, 0::2] = np.sin(position_embeddings_np[:, 0::2])
            position_embeddings_np[:, 1::2] = np.cos(position_embeddings_np[:, 1::2])
            initializer = tf.constant_initializer(position_embeddings_np)

        position_embeddings = tf.get_variable(
            "position_embeddings",
            dtype=tf.float32,
            shape=[max_sequence_length, emb_dim],
            regularizer=tf.contrib.layers.l2_regularizer(0.0) if trainable else None,
            initializer=initializer,
            trainable=trainable,
        )

        batch_size, sequence_length = tf.shape(tensors)[0], tf.shape(tensors)[1]
        position_indices = tf.tile(tf.expand_dims(tf.range(sequence_length), 0), [batch_size, 1])
        return tensors + tf.nn.embedding_lookup(position_embeddings, position_indices)


@base.layer(n_in=1, n_out=1)
def AttentionMask(tensors: tf.Tensor, use_look_ahead_mask: bool):
    """Compute Attention Mask.

    Parameters
    ----------
    tensors : tf.Tensor
        Shape = [batch_size, sequence_length]
    use_look_ahead_mask : bool
        Add look ahead mask if True

    Returns
    -------
    tf.Tensor
        Shape = [batch_size, sequence_length, sequence_length]
    """
    t1 = tf.expand_dims(tensors, axis=-1)
    t2 = tf.expand_dims(tensors, axis=-2)
    attention_mask = tf.logical_and(t1, t2)

    if not use_look_ahead_mask:
        return attention_mask

    sequence_length = tf.shape(tensors)[1]
    sub_diag_ones = tf.linalg.band_part(tf.ones((sequence_length, sequence_length), dtype=tf.bool), -1, 0)
    sub_diag_ones = tf.expand_dims(sub_diag_ones, axis=0)
    return tf.logical_and(attention_mask, sub_diag_ones)


class SelfMultiheadAttention(base.Layer):
    """Self MultiHead Attention Layer.

    Attributes
    ----------
    block_id : int
        Id of the block (scope TF variables using that name)
    dim_head : int
        Dimension of each head
    num_heads : int
        Number of heads
    residual_connection : bool
        If True, add input to output (residual connection)
    """

    def __init__(
        self, num_heads: int, dim_head: int, residual_connection: bool, **kwargs,
    ):
        super().__init__(n_in=2, n_out=1, **kwargs)
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.residual_connection = residual_connection

    def forward(self, tensors: Tuple[tf.Tensor, tf.Tensor], mode: str = None):  # type: ignore
        """Compute MultiHead Attention.

        Parameters
        ----------
        tensors : Tuple[tf.Tensor, tf.Tensor]
            x = [batch_size, sequence_length, dim]
            mask = [batch_size, sequence_length, sequence_length]

        Returns
        -------
        tf.Tensor
            [batch_size, sequence_length, dim]
        """
        # pylint: disable=unused-argument
        # Unpack inputs and retrieve input dimension
        x, mask = tensors
        dim = x.get_shape().as_list()[-1]

        with tf.variable_scope("multihead_attention"):
            # Shape = [batch_size, sequence_length, d_model]
            query = tf.layers.dense(x, self.num_heads * self.dim_head, use_bias=False, name="query")
            key = tf.layers.dense(x, self.num_heads * self.dim_head, use_bias=False, name="key")
            value = tf.layers.dense(x, self.num_heads * self.dim_head, use_bias=False, name="values")

            # Shape = [batch_size, num_heads, sequence_length, dim_head]
            query_heads = self.split_heads(query)
            key_heads = self.split_heads(key)
            value_heads = self.split_heads(value)

            # Shape = [batch_size, num_heads, sequence_length, dim_head]
            scaled_attention = self.scaled_dot_attention(query_heads, key_heads, value_heads, mask)

            # Shape = [batch_size, sequence_length, num_heads * dim_head]
            outputs = self.join_heads(scaled_attention)

            # Shape = [batch_size, sequence_length, dim]
            outputs = tf.layers.dense(outputs, dim)

            if self.residual_connection:
                outputs += x

            return outputs

    def split_heads(self, x):
        """Split the last dimension into heads."""
        batch_size = tf.shape(x)[0]
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.dim_head))
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        return x

    def join_heads(self, x):
        """Join head split tensor (Inverse of split_heads)."""
        batch_size = tf.shape(x)[0]
        x = tf.transpose(x, perm=[0, 2, 1, 3])
        x = tf.reshape(x, (batch_size, -1, self.num_heads * self.dim_head))
        return x

    def scaled_dot_attention(self, query: tf.Tensor, key: tf.Tensor, value: tf.Tensor, mask: tf.Tensor = None):
        """Compute Scaled Dot Attention.

        Parameters
        ----------
        query : tf.Tensor
            Shape = [batch, num_heads, sequence_length, dim_head]
        key : tf.Tensor
            Shape = [batch, num_heads, sequence_length, dim_head]
        value : tf.Tensor
            Shape = [batch, num_heads, sequence_length, dim_head]
        mask : tf.Tensor, optional
            Shape = [batch, sequence_length, sequence_length]

        Returns
        -------
        tf.Tensor
            shape = [batch, heads, sequence_length, d]
        """
        # Shape = [batch, num_heads, sequence_length, sequence_length]
        scores = tf.matmul(query, key, transpose_b=True)
        scores /= tf.math.sqrt(float(self.dim_head))

        # Set masked scores to -inf before softmax
        if mask is not None:
            not_mask = tf.logical_not(mask)
            not_mask_float = tf.cast(not_mask, tf.float32)
            scores += tf.expand_dims(not_mask_float, axis=1) * -1e9

        # Shape = [batch, num_heads, sequence_length, sequence_length]
        attention_weights = tf.nn.softmax(scores, axis=-1)

        if mask is not None:
            attention_weights *= tf.expand_dims(tf.cast(mask, tf.float32), axis=1)

        # Shape = [batch, num_heads, sequence_length, dim_head]
        attention_value = tf.matmul(attention_weights, value)
        return attention_value
