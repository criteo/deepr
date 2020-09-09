# pylint: disable=no-value-for-parameter,invalid-name,unexpected-keyword-arg
"""VAE Model."""

import logging
from typing import Tuple

import tensorflow as tf
import deepr as dpr


LOGGER = logging.getLogger(__name__)


def MultiVAELoss(
    vocab_size: int,
    dim: int,
    embeddings_name: str,
    reuse_embeddings: bool,
    beta_start: float,
    beta_end: float,
    beta_steps: int,
    reg: float,
):
    """Compute Multi-VAE loss."""
    return dpr.layers.Sequential(
        dpr.layers.Select(inputs=("userEmbeddings", "KL", "one_hot")),
        Logits(
            inputs="userEmbeddings",
            outputs="logits",
            vocab_size=vocab_size,
            dim=dim,
            embeddings_name=embeddings_name,
            reuse_embeddings=reuse_embeddings,
        ),
        MultiLoss(inputs=("logits", "one_hot"), outputs="loss_multi"),
        AddKLandL2(
            inputs=("loss_multi", "KL"),
            outputs=("loss", "beta", "l2"),
            beta_start=beta_start,
            beta_end=beta_end,
            beta_steps=beta_steps,
            reg=reg,
        ),
        dpr.layers.Select(
            inputs=("loss", "loss_multi", "KL", "l2", "beta"),
            outputs=("loss", "loss_multi", "loss_KL", "loss_l2", "beta"),
        ),
    )


@dpr.layers.layer(n_in=1, n_out=1)
def Logits(tensors: tf.Tensor, vocab_size: int, dim: int, embeddings_name: str, reuse_embeddings: bool):
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_embeddings):
        embeddings = tf.get_variable(name=embeddings_name, shape=(vocab_size, dim))

    biases = tf.get_variable(name="biases", shape=(vocab_size,), initializer=tf.zeros_initializer())
    logits = tf.linalg.matmul(tensors, embeddings, transpose_b=True) + tf.expand_dims(biases, axis=0)
    return logits


@dpr.layers.layer(n_in=2, n_out=1)
def MultiLoss(tensors: Tuple[tf.Tensor]):
    """Multinomial loss."""
    logits, one_hot = tensors
    log_softmax = tf.nn.log_softmax(logits)
    return -tf.reduce_mean(tf.reduce_sum(log_softmax * tf.cast(one_hot, tf.float32), axis=-1))


@dpr.layers.layer(n_in=2, n_out=3)
def AddKLandL2(tensors: Tuple[tf.Tensor], beta_start: float, beta_end: float, beta_steps: int, reg: float = None):
    """Add KL loss, decay beta linearly during training + L2 reg."""
    loss, KL = tensors

    # Linearly decay beta from start to end for steps
    beta = tf.train.polynomial_decay(
        float(beta_start), tf.train.get_global_step(), beta_steps, float(beta_end), power=1.0, cycle=False
    )
    # Combine loss with KL divergence
    loss = loss + beta * KL

    # Add L2 regularization
    l2 = tf.constant(0.0, dtype=tf.float32)
    if reg:
        l2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
        loss += 2 * reg * l2

    return loss, beta, l2


def VAEModel(
    vocab_size: int,
    dim: int,
    dims_encode: Tuple[int] = (200,),
    dims_decode: Tuple[int] = (200,),
    activation=tf.nn.tanh,
    embeddings_name: str = "embeddings",
    keep_prob: float = 0.5,
) -> dpr.layers.Layer:
    """VAE Model."""
    return dpr.layers.Sequential(
        dpr.layers.Select(inputs=("inputPositives", "inputMask")),
        RandomMask(inputs="inputMask", outputs="inputMask", keep_prob=keep_prob),
        dpr.layers.Embedding(
            inputs="inputPositives", outputs="inputEmbeddings", variable_name=embeddings_name, shape=(vocab_size, dim)
        ),
        dpr.layers.ToFloat(inputs="inputMask", outputs="inputWeights"),
        AverageAndBias(inputs=("inputEmbeddings", "inputWeights"), outputs="averageEmbeddings"),
        Activation(inputs="averageEmbeddings", outputs="averageEmbeddings", activation=activation),
        Encode(inputs="averageEmbeddings", outputs=("mu", "std", "KL"), dims=dims_encode, activation=activation),
        GaussianNoise(inputs=("mu", "std"), outputs="latent"),
        Decode(inputs="latent", outputs="userEmbeddings", dims=dims_decode, activation=activation),
        dpr.layers.Select(inputs=("userEmbeddings", "KL"), outputs=("userEmbeddings", "KL")),
    )


@dpr.layers.layer(n_in=1, n_out=1)
def RandomMask(tensors: tf.Tensor, mode: str, keep_prob: float):
    if mode == dpr.TRAIN:
        LOGGER.info("Applying random mask to inputs")
        mask = tf.random.uniform(tf.shape(tensors)) <= keep_prob
        return tf.logical_and(tensors, mask)
    return tensors


@dpr.layers.layer(n_in=2, n_out=1)
def AverageAndBias(tensors: Tuple[tf.Tensor]):
    vectors, weights = tensors
    dim = vectors.shape[-1]
    vectors *= tf.expand_dims(weights, axis=-1)
    average = tf.div_no_nan(
        tf.reduce_sum(vectors, axis=-2), tf.expand_dims(tf.sqrt(tf.reduce_sum(weights, axis=-1)), axis=-1)
    )
    biases = tf.get_variable(name="embeddings_biases", shape=(dim,))
    return average + tf.expand_dims(biases, axis=0)


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
        mu = tensors[:, : dims[-1]]
        logvar = tensors[:, dims[-1] :]
        std = tf.exp(0.5 * logvar)

        # KL divergence with standard normal
        KL = tf.reduce_mean(tf.reduce_sum(0.5 * (-logvar + tf.exp(logvar) + mu ** 2 - 1), axis=1))

        return mu, std, KL


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
    if mode == dpr.TRAIN:
        LOGGER.info("Sampling latent variable (TRAIN only).")
        epsilon = tf.random_normal(tf.shape(std))
        return mu + std * epsilon
    return mu
