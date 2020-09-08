# pylint: disable=no-value-for-parameter,invalid-name,unexpected-keyword-arg
"""VAE Model."""

import logging
from typing import Tuple

import tensorflow as tf
import deepr as dpr


LOGGER = logging.getLogger(__name__)


def MultiVAELoss(vocab_size: int, dim: int, embeddings_name: str, reuse_embeddings: bool, beta: float):
    return dpr.layers.Sequential(
        dpr.layers.Select(inputs=("userEmbeddings", "KL", "inputPositives", "inputMask")),
        Logits(
            inputs="userEmbeddings",
            outputs="logits",
            vocab_size=vocab_size,
            dim=dim,
            embeddings_name=embeddings_name,
            reuse_embeddings=reuse_embeddings,
        ),
        MultiLoss(inputs=("logits", "inputPositives", "inputMask"), outputs="loss_multi", vocab_size=vocab_size),
        AddKL(inputs=("loss_multi", "KL"), outputs="loss", beta=beta),
        dpr.layers.Select(inputs=("loss", "loss_multi", "KL"), outputs=("loss", "loss_multi", "loss_KL")),
    )


@dpr.layers.layer(n_in=1, n_out=1)
def Logits(tensors: tf.Tensor, vocab_size: int, dim: int, embeddings_name: str, reuse_embeddings: bool):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_embeddings):
        embeddings = tf.get_variable(name=embeddings_name, shape=(vocab_size, dim))

    biases = tf.get_variable(name="biases", shape=(vocab_size,), initializer=tf.zeros_initializer())
    logits = tf.linalg.matmul(tensors, embeddings, transpose_b=True) + tf.expand_dims(biases, axis=0)
    return logits


@dpr.layers.layer(n_in=3, n_out=1)
def MultiLoss(tensors: Tuple[tf.Tensor], vocab_size: int):
    """Multinomial loss."""
    logits, indices, mask = tensors

    # Convert indices to one hot [batch, sequence_length, vocab_size]
    one_hots = tf.one_hot(indices, depth=vocab_size, dtype=tf.float32)
    one_hots *= tf.expand_dims(tf.cast(mask, tf.float32), axis=-1)
    one_hot = tf.reduce_sum(one_hots, axis=1)

    # Compute log softmax for each one hot indices
    log_softmax = tf.nn.log_softmax(logits)
    return -tf.reduce_mean(tf.reduce_sum(log_softmax * one_hot, axis=-1))


@dpr.layers.layer(n_in=2, n_out=1)
def AddKL(tensors: Tuple[tf.Tensor], beta: float):
    loss, KL = tensors
    return loss + beta * KL


def VAEModel(
    vocab_size: int,
    dim: int,
    dims_encode: Tuple[int] = (200,),
    dims_decode: Tuple[int] = (200,),
    activation=tf.nn.tanh,
    embeddings_name: str = "embeddings",
) -> dpr.layers.Layer:
    """VAE Model."""
    return dpr.layers.Sequential(
        dpr.layers.Select(inputs=("inputPositives", "inputMask")),
        dpr.layers.Embedding(
            inputs="inputPositives", outputs="inputEmbeddings", variable_name=embeddings_name, shape=(vocab_size, dim)
        ),
        dpr.layers.ToFloat(inputs="inputMask", outputs="inputWeights"),
        dpr.layers.WeightedAverage(inputs=("inputEmbeddings", "inputWeights"), outputs="averageEmbeddings"),
        Activation(inputs="averageEmbeddings", outputs="averageEmbeddings", activation=activation),
        Encode(inputs="averageEmbeddings", outputs=("mu", "std", "KL"), dims=dims_encode, activation=activation),
        GaussianNoise(inputs=("mu", "std"), outputs="latent"),
        Decode(inputs="latent", outputs="userEmbeddings", dims=dims_decode, activation=activation),
        dpr.layers.Select(inputs=("userEmbeddings", "KL"), outputs=("userEmbeddings", "KL")),
    )


@dpr.layers.layer(n_in=1, n_out=1)
def Activation(tensors: tf.Tensor, activation=tf.nn.tanh):
    return activation(tensors)


@dpr.layers.layer(n_in=1, n_out=3)
def Encode(tensors: tf.Tensor, dims: Tuple, activation=tf.nn.tanh):
    """Encode tensor, apply KL constraint."""
    with tf.variable_scope("encoder"):
        # Hidden layers (dense layers with tanh activation)
        for dim in dims[:-1]:
            tensors = tf.layers.dense(inputs=tensors, units=dim, activation=activation)

        # Last layer predicts mean and log variance of the latent user
        tensors = tf.layers.dense(inputs=tensors, units=2 * dims[-1], activation=None)

        # Predict prior statistics
        mu_q = tensors[:, : dims[-1]]
        logvar_q = tensors[:, dims[-1] :]
        std_q = tf.exp(0.5 * logvar_q)

        # KL divergence with standard normal
        KL = tf.reduce_mean(tf.reduce_sum(0.5 * (-logvar_q + tf.exp(logvar_q) + mu_q ** 2 - 1), axis=1))

        return mu_q, std_q, KL


@dpr.layers.layer(n_in=1, n_out=1)
def Decode(tensors: tf.Tensor, dims: Tuple, activation=tf.nn.tanh):
    """Decode tensor."""
    with tf.variable_scope("decoder"):
        for dim in dims:
            tensors = tf.layers.dense(inputs=tensors, units=dim, activation=activation)
    return tensors


@dpr.layers.layer(n_in=2, n_out=1)
def GaussianNoise(tensors: Tuple[tf.Tensor], mode: str = None):
    mu, std = tensors
    epsilon = tf.random_normal(tf.shape(std))
    if mode == dpr.TRAIN:
        LOGGER.info("Sampling latent variable (TRAIN only).")
        return mu + std * epsilon
    return mu
