# pylint: disable=no-value-for-parameter,invalid-name,unexpected-keyword-arg
"""VAE Model."""

import logging
from typing import Tuple

import tensorflow as tf
import deepr as dpr


LOGGER = logging.getLogger(__name__)


def VAEModel(
    vocab_size: int,
    dims_encode: Tuple[int, ...] = (600, 200),
    dims_decode: Tuple[int, ...] = (200, 600),
    keep_prob: float = 0.5,
    train_embeddings: bool = True,
    project: bool = False,
    share_embeddings: bool = False,
    seed: int = 42,
) -> dpr.layers.Layer:
    """VAE Model."""
    if dims_encode[-1] != dims_decode[0]:
        msg = f"Encoder's latent dim != decoder's ({dims_encode[-1]} != {dims_decode[0]})"
        raise ValueError(msg)
    return dpr.layers.Sequential(
        dpr.layers.Select(inputs=("inputPositives", "inputMask")),
        dpr.layers.Embedding(
            inputs="inputPositives",
            outputs="inputEmbeddings",
            variable_name="embeddings" if share_embeddings else "encoder/embeddings",
            shape=(vocab_size, dims_encode[0]),
            trainable=train_embeddings,
            initializer=tf.contrib.layers.xavier_initializer(seed=seed),
        ),
        WeightedSum(
            inputs=("inputEmbeddings", "inputMask"), outputs="averageEmbeddings", keep_prob=keep_prob, seed=seed
        ),
        Projection(
            inputs="averageEmbeddings", outputs="averageEmbeddings", variable_name="encoder/projection", seed=seed
        )
        if project
        else [],
        AddBias(inputs="averageEmbeddings", outputs="averageEmbeddings", variable_name="encoder/bias", seed=seed),
        dpr.layers.Lambda(
            lambda tensors, _: tf.nn.tanh(tensors), inputs="averageEmbeddings", outputs="averageEmbeddings"
        ),
        Encode(
            inputs="averageEmbeddings",
            outputs=("mu", "std", "KL"),
            dims=dims_encode[1:],
            activation=tf.nn.tanh,
            seed=seed,
        ),
        GaussianNoise(inputs=("mu", "std"), outputs="latent", seed=seed),
        Decode(inputs="latent", outputs="userEmbeddings", dims=dims_decode[1:], activation=tf.nn.tanh, seed=seed),
        Projection(inputs="userEmbeddings", outputs="userEmbeddings", variable_name="decoder/projection", seed=seed)
        if project
        else [],
        Logits(
            inputs="userEmbeddings",
            outputs="logits",
            vocab_size=vocab_size,
            dim=dims_decode[-1],
            trainable=train_embeddings,
            reuse=share_embeddings,
            seed=seed,
        ),
        dpr.layers.Select(inputs=("userEmbeddings", "logits", "mu", "std", "KL")),
    )


@dpr.layers.layer(n_in=2, n_out=1)
def WeightedSum(tensors: tf.Tensor, mode: str, keep_prob: float, seed: int):
    """Compute Weighted Sum (randomly masking inputs in TRAIN mode)."""
    embeddings, mask = tensors
    weights = tf.cast(mask, tf.float32)
    weights = tf.nn.l2_normalize(weights, axis=-1)
    if mode == dpr.TRAIN:
        LOGGER.info("Applying random mask to inputs (TRAIN only)")
        weights = tf.nn.dropout(weights, keep_prob=keep_prob, seed=seed)

    embeddings *= tf.expand_dims(weights, axis=-1)
    return tf.reduce_sum(embeddings, axis=-2)


@dpr.layers.layer(n_in=1, n_out=1)
def Projection(tensors: tf.Tensor, variable_name: str, seed: int):
    """Apply symmetric transform to non-projected user embeddings."""
    # Resolve embeddings dimension
    dim = int(tensors.shape[-1])
    if not isinstance(dim, int):
        raise TypeError(f"Expected static shape for {tensors} but got {dim} (must be INT)")

    # Retrieve projection matrix
    with tf.variable_scope(tf.get_variable_scope(), reuse=False):
        projection_matrix = tf.get_variable(
            name=variable_name, shape=[dim, dim], initializer=tf.contrib.layers.xavier_initializer(seed=seed)
        )

    return tf.matmul(tensors, projection_matrix)


@dpr.layers.layer(n_in=1, n_out=1)
def AddBias(tensors: tf.Tensor, variable_name: str, seed: int):
    dim = tensors.shape[-1]
    biases = tf.get_variable(
        name=variable_name, shape=(dim,), initializer=tf.truncated_normal_initializer(stddev=0.001, seed=seed)
    )
    return tensors + tf.expand_dims(biases, axis=0)


@dpr.layers.layer(n_in=1, n_out=3)
def Encode(tensors: tf.Tensor, dims: Tuple, activation, seed: int):
    """Encode tensor, apply KL constraint."""
    with tf.variable_scope("encoder"):
        # Hidden layers
        for dim in dims[:-1]:
            tensors = tf.layers.dense(
                inputs=tensors,
                units=dim,
                activation=activation,
                kernel_initializer=tf.contrib.layers.xavier_initializer(seed=seed),
                bias_initializer=tf.truncated_normal_initializer(stddev=0.001, seed=seed),
            )

        # Last layer predicts mean and log variance of the latent user
        tensors = tf.layers.dense(
            inputs=tensors,
            units=2 * dims[-1],
            activation=None,
            kernel_initializer=tf.contrib.layers.xavier_initializer(seed=seed),
            bias_initializer=tf.truncated_normal_initializer(stddev=0.001, seed=seed),
        )

        # Predict prior statistics
        mu = tensors[:, : dims[-1]]
        logvar = tensors[:, dims[-1] :]
        std = tf.exp(0.5 * logvar)

        # KL divergence with standard normal
        KL = tf.reduce_mean(tf.reduce_sum(0.5 * (-logvar + tf.exp(logvar) + mu ** 2 - 1), axis=1))

        return mu, std, KL


@dpr.layers.layer(n_in=2, n_out=1)
def GaussianNoise(tensors: Tuple[tf.Tensor, tf.Tensor], mode: str, seed: int):
    mu, std = tensors
    if mode == dpr.TRAIN:
        LOGGER.info("Sampling latent variable (TRAIN only).")
        epsilon = tf.random_normal(tf.shape(std), seed=seed)
        return mu + std * epsilon
    return mu


@dpr.layers.layer(n_in=1, n_out=1)
def Decode(tensors: tf.Tensor, dims: Tuple, activation, seed: int):
    """Decode tensor."""
    with tf.variable_scope("decoder"):
        for dim in dims:
            tensors = tf.layers.dense(
                inputs=tensors,
                units=dim,
                activation=activation,
                kernel_initializer=tf.contrib.layers.xavier_initializer(seed=seed),
                bias_initializer=tf.truncated_normal_initializer(stddev=0.001, seed=seed),
            )
    return tensors


@dpr.layers.layer(n_in=1, n_out=1)
def Logits(tensors: tf.Tensor, vocab_size: int, dim: int, trainable: bool, reuse: bool, seed: int):
    """Compute logits given user and create target item embeddings."""
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
        embeddings = tf.get_variable(
            name="embeddings",
            shape=(vocab_size, dim),
            trainable=trainable,
            initializer=tf.contrib.layers.xavier_initializer(seed=seed),
        )
    biases = tf.get_variable(
        name="biases", shape=(vocab_size,), initializer=tf.truncated_normal_initializer(stddev=0.001, seed=seed)
    )
    logits = tf.linalg.matmul(tensors, embeddings, transpose_b=True) + tf.expand_dims(biases, axis=0)
    return logits
