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
) -> dpr.layers.Layer:
    """VAE Model."""
    if dims_encode[-1] != dims_decode[0]:
        msg = f"Encoder's latent dim != decoder's ({dims_encode[-1]} != {dims_decode[0]})"
        raise ValueError(msg)
    return dpr.layers.Sequential(
        dpr.layers.Select(inputs=("inputPositives", "inputMask")),
        RandomMask(inputs="inputMask", outputs="inputMask", keep_prob=keep_prob),
        dpr.layers.Embedding(
            inputs="inputPositives",
            outputs="inputEmbeddings",
            variable_name="encoder/embeddings",
            shape=(vocab_size, dims_encode[0]),
        ),
        dpr.layers.ToFloat(inputs="inputMask", outputs="inputWeights"),
        Average(inputs=("inputEmbeddings", "inputWeights"), outputs="averageEmbeddings"),
        AddBias(inputs="averageEmbeddings", outputs="averageEmbeddings", variable_name="encoder/bias"),
        dpr.layers.Lambda(
            lambda tensors, _: tf.nn.tanh(tensors), inputs="averageEmbeddings", outputs="averageEmbeddings"
        ),
        Encode(inputs="averageEmbeddings", outputs=("mu", "std", "KL"), dims=dims_encode[1:]),
        GaussianNoise(inputs=("mu", "std"), outputs="latent"),
        Decode(inputs="latent", outputs="userEmbeddings", dims=dims_decode[1:]),
        Logits(inputs="userEmbeddings", outputs="logits", vocab_size=vocab_size, dim=dims_decode[-1]),
        dpr.layers.Select(inputs=("userEmbeddings", "logits", "mu", "std", "KL")),
    )


@dpr.layers.layer(n_in=1, n_out=1)
def RandomMask(tensors: tf.Tensor, mode: str, keep_prob: float):
    if mode == dpr.TRAIN:
        LOGGER.info("Applying random mask to inputs (TRAIN only)")
        mask = tf.random.uniform(tf.shape(tensors)) <= keep_prob
        return tf.logical_and(tensors, mask)
    return tensors


@dpr.layers.layer(n_in=2, n_out=1)
def Average(tensors: Tuple[tf.Tensor, tf.Tensor]):
    vectors, weights = tensors
    vectors *= tf.expand_dims(weights, axis=-1)
    return tf.div_no_nan(tf.reduce_sum(vectors, axis=-2), tf.sqrt(tf.reduce_sum(weights, axis=-1, keepdims=True)))


@dpr.layers.layer(n_in=1, n_out=1)
def AddBias(tensors: tf.Tensor, variable_name: str):
    dim = tensors.shape[-1]
    biases = tf.get_variable(name=variable_name, shape=(dim,), initializer=tf.zeros_initializer())
    return tensors + tf.expand_dims(biases, axis=0)


@dpr.layers.layer(n_in=1, n_out=3)
def Encode(tensors: tf.Tensor, dims: Tuple, activation=tf.nn.tanh):
    """Encode tensor, apply KL constraint."""
    with tf.variable_scope("encoder"):
        # Hidden layers
        for dim in dims[:-1]:
            tensors = tf.layers.dense(inputs=tensors, units=dim, activation=activation)

        # Last layer predicts mean and log variance of the latent user
        tensors = tf.layers.dense(inputs=tensors, units=2 * dims[-1], activation=None)

        # Predict prior statistics
        mu = tensors[:, : dims[-1]]
        logvar = tensors[:, dims[-1] :]
        std = tf.exp(0.5 * logvar)

        # KL divergence with standard normal
        KL = tf.reduce_mean(tf.reduce_sum(0.5 * (-logvar + tf.exp(logvar) + mu ** 2 - 1), axis=1))

        return mu, std, KL


@dpr.layers.layer(n_in=2, n_out=1)
def GaussianNoise(tensors: Tuple[tf.Tensor, tf.Tensor], mode: str = None):
    mu, std = tensors
    if mode == dpr.TRAIN:
        LOGGER.info("Sampling latent variable (TRAIN only).")
        epsilon = tf.random_normal(tf.shape(std))
        return mu + std * epsilon
    return mu


@dpr.layers.layer(n_in=1, n_out=1)
def Decode(tensors: tf.Tensor, dims: Tuple, activation=tf.nn.tanh):
    """Decode tensor."""
    with tf.variable_scope("decoder"):
        for dim in dims:
            tensors = tf.layers.dense(inputs=tensors, units=dim, activation=activation)
    return tensors


@dpr.layers.layer(n_in=1, n_out=1)
def Logits(tensors: tf.Tensor, vocab_size: int, dim: int):
    embeddings = tf.get_variable(name="embeddings", shape=(vocab_size, dim))
    biases = tf.get_variable(name="biases", shape=(vocab_size,), initializer=tf.zeros_initializer())
    logits = tf.linalg.matmul(tensors, embeddings, transpose_b=True) + tf.expand_dims(biases, axis=0)
    return logits
