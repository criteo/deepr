# pylint: disable=unexpected-keyword-arg,no-value-for-parameter,invalid-name
"""Average Model."""

import logging

import tensorflow as tf
import deepr as dpr


LOGGER = logging.getLogger(__name__)


def AverageModel(vocab_size: int, dim: int, keep_prob: float, train_embeddings: bool = True, project: bool = False):
    """Average Model."""
    return dpr.layers.Sequential(
        dpr.layers.Select(inputs=("inputPositives", "inputMask")),
        RandomMask(inputs="inputMask", outputs="inputMask", keep_prob=keep_prob),
        dpr.layers.Embedding(
            inputs="inputPositives",
            outputs="inputEmbeddings",
            variable_name="embeddings",
            shape=(vocab_size, dim),
            trainable=train_embeddings,
        ),
        dpr.layers.ToFloat(inputs="inputMask", outputs="inputWeights"),
        dpr.layers.WeightedAverage(inputs=("inputEmbeddings", "inputWeights"), outputs="userEmbeddings"),
        Projection(inputs="userEmbeddings", outputs="userEmbeddings") if project else [],
        Logits(inputs="userEmbeddings", outputs="logits", vocab_size=vocab_size, dim=dim),
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
def Projection(tensors: tf.Tensor):
    """Apply symmetric transform to non-projected user embeddings."""
    # Resolve embeddings dimension
    dim = int(tensors.shape[-1])
    if not isinstance(dim, int):
        raise TypeError(f"Expected static shape for {tensors} but got {dim} (must be INT)")

    # Retrieve projection matrix
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        projection_matrix = tf.get_variable(name="projection_matrix", shape=[dim, dim])

    # During training, applies symmetric transform
    S = tf.matmul(projection_matrix, projection_matrix, transpose_b=True)
    return tf.matmul(tensors, S)


@dpr.layers.layer(n_in=1, n_out=1)
def Logits(tensors: tf.Tensor, vocab_size: int, dim: int):
    with tf.variable_scope(tf.get_variable_scope(), reuse=True):
        embeddings = tf.get_variable(name="embeddings", shape=(vocab_size, dim))
    biases = tf.get_variable(
        name="biases", shape=(vocab_size,), initializer=tf.truncated_normal_initializer(stddev=0.001)
    )
    logits = tf.linalg.matmul(tensors, embeddings, transpose_b=True) + tf.expand_dims(biases, axis=0)
    return logits
