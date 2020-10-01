# pylint: disable=unexpected-keyword-arg,no-value-for-parameter,invalid-name
"""Average Model."""

import logging

import tensorflow as tf
import deepr as dpr


LOGGER = logging.getLogger(__name__)


def AverageModel(
    vocab_size: int,
    dim: int,
    keep_prob: float,
    share_embeddings: bool = True,
    train_embeddings: bool = True,
    average_with_bias: bool = False,
    project: bool = False,
):
    """Average Model."""
    return dpr.layers.Sequential(
        dpr.layers.Select(inputs=("inputPositives", "inputMask")),
        RandomMask(inputs="inputMask", outputs="inputMask", keep_prob=keep_prob),
        dpr.layers.Embedding(
            inputs="inputPositives",
            outputs="inputEmbeddings",
            variable_name="embeddings" if share_embeddings else "encoder/embeddings",
            shape=(vocab_size, dim),
            trainable=train_embeddings,
        ),
        dpr.layers.ToFloat(inputs="inputMask", outputs="inputWeights"),
        dpr.layers.WeightedAverage(inputs=("inputEmbeddings", "inputWeights"), outputs="userEmbeddings"),
        Projection(
            inputs="userEmbeddings",
            outputs="userEmbeddings",
            name="projection" if share_embeddings else "encoder/projection",
            reuse=False,
            transpose=False,
        )
        if project
        else [],
        AddBias(inputs="userEmbeddings", outputs="userEmbeddings") if average_with_bias else [],
        Projection(
            inputs="userEmbeddings",
            outputs="userEmbeddings",
            name="projection",
            reuse=share_embeddings,
            transpose=True,
        )
        if project
        else [],
        Logits(inputs="userEmbeddings", outputs="logits", vocab_size=vocab_size, dim=dim, reuse=share_embeddings),
        dpr.layers.Select(inputs=("userEmbeddings", "logits")),
    )


@dpr.layers.layer(n_in=1, n_out=1)
def RandomMask(tensors: tf.Tensor, mode: str, keep_prob: float):
    if mode == dpr.TRAIN and keep_prob is not None:
        LOGGER.info("Applying random mask to inputs (TRAIN only)")
        mask = tf.random.uniform(tf.shape(tensors)) <= keep_prob
        return tf.logical_and(tensors, mask)
    return tensors


@dpr.layers.layer(n_in=1, n_out=1)
def AddBias(tensors: tf.Tensor):
    dim = tensors.shape[-1]
    biases = tf.get_variable(
        name="encoder/biases", shape=(dim,), initializer=tf.truncated_normal_initializer(stddev=0.001)
    )
    return tensors + tf.expand_dims(biases, axis=0)


@dpr.layers.layer(n_in=1, n_out=1)
def Projection(tensors: tf.Tensor, name: str, reuse: bool = False, transpose: bool = False):
    """Apply symmetric transform to non-projected user embeddings."""
    dim = int(tensors.shape[-1])
    if not isinstance(dim, int):
        raise TypeError(f"Expected static shape for {tensors} but got {dim} (must be INT)")
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
        projection_matrix = tf.get_variable(name=name, shape=[dim, dim])

    return tf.matmul(tensors, projection_matrix, transpose_b=transpose)


@dpr.layers.layer(n_in=1, n_out=1)
def Logits(tensors: tf.Tensor, vocab_size: int, dim: int, reuse: bool = True):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
        embeddings = tf.get_variable(name="embeddings", shape=(vocab_size, dim))
    biases = tf.get_variable(
        name="biases", shape=(vocab_size,), initializer=tf.truncated_normal_initializer(stddev=0.001)
    )
    logits = tf.linalg.matmul(tensors, embeddings, transpose_b=True) + tf.expand_dims(biases, axis=0)
    return logits
