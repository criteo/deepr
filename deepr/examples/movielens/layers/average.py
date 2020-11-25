# pylint: disable=unexpected-keyword-arg,no-value-for-parameter,invalid-name
"""Average Model."""

import logging

import tensorflow as tf
import deepr


LOGGER = logging.getLogger(__name__)


def AverageModel(
    vocab_size: int,
    dim: int,
    keep_prob: float,
    share_embeddings: bool = True,
    train_embeddings: bool = True,
    train_biases: bool = True,
    average_with_bias: bool = False,
    project: bool = False,
    reduce_mode: str = "average",
):
    """Average Model."""
    return deepr.layers.DAG(
        deepr.layers.Select(inputs=("inputPositives", "inputMask")),
        deepr.layers.Embedding(
            inputs="inputPositives",
            outputs="inputEmbeddings",
            variable_name="embeddings" if share_embeddings else "encoder/embeddings",
            shape=(vocab_size, dim),
            trainable=train_embeddings,
        ),
        UserEmbedding(
            inputs=("inputEmbeddings", "inputMask"),
            outputs="userEmbeddings",
            keep_prob=keep_prob,
            reduce_mode=reduce_mode,
        ),
        Projection(
            inputs="userEmbeddings",
            outputs="userEmbeddings",
            variable_name="projection" if share_embeddings else "encoder/projection",
            reuse=False,
            transpose=False,
        )
        if project
        else [],
        AddBias(inputs="userEmbeddings", outputs="userEmbeddings") if average_with_bias else [],
        Projection(
            inputs="userEmbeddings",
            outputs="userEmbeddings",
            variable_name="projection",
            reuse=share_embeddings,
            transpose=True,
        )
        if project
        else [],
        Logits(
            inputs="userEmbeddings",
            outputs="logits",
            vocab_size=vocab_size,
            dim=dim,
            reuse=share_embeddings,
            trainable=train_biases,
        ),
        deepr.layers.Select(inputs=("userEmbeddings", "logits")),
    )


@deepr.layers.layer(n_in=2, n_out=1)
def UserEmbedding(tensors: tf.Tensor, mode: str, keep_prob: float, reduce_mode: str = "average"):
    """Compute Weighted Sum (randomly masking inputs in TRAIN mode)."""
    embeddings, mask = tensors

    # Drop entries without re-scaling (not classical dropout)
    if mode == deepr.TRAIN:
        LOGGER.info("Applying random mask to inputs (TRAIN only)")
        mask_random = tf.random.uniform(tf.shape(mask)) <= keep_prob
        mask = tf.logical_and(mask, mask_random)

    weights = tf.cast(mask, tf.float32)

    # Scale the weights depending on the reduce mode
    if reduce_mode == "l2":
        weights = tf.nn.l2_normalize(weights, axis=-1)
    elif reduce_mode == "average":
        weights = tf.div_no_nan(weights, tf.reduce_sum(weights, axis=-1, keepdims=True))
    elif reduce_mode == "sum":
        pass
    else:
        raise ValueError(f"Reduce mode {reduce_mode} unknown (must be 'l2', 'average' or 'sum')")

    return tf.reduce_sum(embeddings * tf.expand_dims(weights, axis=-1), axis=-2)


@deepr.layers.layer(n_in=1, n_out=1)
def AddBias(tensors: tf.Tensor):
    dim = tensors.shape[-1]
    biases = tf.get_variable(
        name="encoder/biases", shape=(dim,), initializer=tf.truncated_normal_initializer(stddev=0.001)
    )
    return tensors + tf.expand_dims(biases, axis=0)


@deepr.layers.layer(n_in=1, n_out=1)
def Projection(tensors: tf.Tensor, variable_name: str, reuse: bool = False, transpose: bool = False):
    """Apply symmetric transform to non-projected user embeddings."""
    dim = int(tensors.shape[-1])
    if not isinstance(dim, int):
        raise TypeError(f"Expected static shape for {tensors} but got {dim} (must be INT)")
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
        projection_matrix = tf.get_variable(name=variable_name, shape=[dim, dim])

    return tf.matmul(tensors, projection_matrix, transpose_b=transpose)


@deepr.layers.layer(n_in=1, n_out=1)
def Logits(tensors: tf.Tensor, vocab_size: int, dim: int, reuse: bool = True, trainable: bool = True):
    """Computes logits as <u, i> + b_i."""
    # Retrieve variables (embeddings and biases)
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
        embeddings = tf.get_variable(name="embeddings", shape=(vocab_size, dim))
    biases = tf.get_variable(
        name="biases",
        shape=(vocab_size,),
        initializer=tf.truncated_normal_initializer(stddev=0.001),
        trainable=trainable,
    )

    # Compute inner product between user and product embeddings
    logits = tf.linalg.matmul(tensors, embeddings, transpose_b=True)

    # Add bias
    logits += tf.expand_dims(biases, axis=0)
    return logits
