# pylint: disable=no-value-for-parameter,invalid-name,unexpected-keyword-arg
"""Multinomial Loss for the Multi-VAE."""

from typing import Tuple
import logging

import tensorflow as tf
import deepr as dpr


LOGGER = logging.getLogger(__name__)


def VAELossSample(beta_start: float, beta_end: float, beta_steps: int, num_negatives: int, vocab_size: int, dim: int):
    """Build Layer for Multi VAE loss with Complementarity Sampling."""
    return dpr.layers.Sequential(
        dpr.layers.Select(inputs=("userEmbeddings", "inputPositives", "inputMask", "KL")),
        RandomNegatives(
            inputs="inputPositives", outputs="inputNegatives", num_negatives=num_negatives, vocab_size=vocab_size,
        ),
        dpr.layers.Embedding(
            inputs="inputPositives",
            outputs="inputPositiveEmbeddings",
            variable_name="embeddings",
            shape=(vocab_size, dim),
            reuse=True,
        ),
        dpr.layers.Embedding(
            inputs="inputNegatives",
            outputs="inputNegativeEmbeddings",
            variable_name="embeddings",
            shape=(vocab_size, dim),
            reuse=True,
        ),
        dpr.layers.Embedding(
            inputs="inputPositives",
            outputs="inputPositiveBiases",
            variable_name="biases",
            shape=(vocab_size,),
            reuse=True,
        ),
        dpr.layers.Embedding(
            inputs="inputNegatives",
            outputs="inputNegativeBiases",
            variable_name="biases",
            shape=(vocab_size,),
            reuse=True,
        ),
        dpr.layers.DotProduct(inputs=("userEmbeddings", "inputPositiveEmbeddings"), outputs="inputPositiveProduct"),
        dpr.layers.DotProduct(inputs=("userEmbeddings", "inputNegativeEmbeddings"), outputs="inputNegativeProduct"),
        dpr.layers.Add(inputs=("inputPositiveProduct", "inputPositiveBiases"), outputs="inputPositiveLogits"),
        dpr.layers.Add(inputs=("inputNegativeProduct", "inputNegativeBiases"), outputs="inputNegativeLogits"),
        SampledMultiLoss(
            inputs=("inputPositiveLogits", "inputNegativeLogits", "inputMask"),
            outputs="loss_multi",
            vocab_size=vocab_size,
            num_negatives=num_negatives,
        ),
        AddKL(
            inputs=("loss_multi", "KL"),
            outputs=("loss", "beta"),
            beta_start=beta_start,
            beta_end=beta_end,
            beta_steps=beta_steps,
        ),
        dpr.layers.Select(inputs=("loss", "loss_multi", "beta")),
    )


@dpr.layers.layer(n_in=1, n_out=1)
def RandomNegatives(tensors, num_negatives, vocab_size):
    return tf.random.uniform(shape=[tf.shape(tensors)[0], num_negatives], maxval=vocab_size, dtype=tf.int64)


@dpr.layers.layer(n_in=3, n_out=1)
def SampledMultiLoss(tensors: Tuple[tf.Tensor, tf.Tensor, tf.Tensor], vocab_size, num_negatives):
    """Sampled Multi loss with Complementary Sum Sampling.

    See http://proceedings.mlr.press/v54/botev17a/botev17a.pdf
    """
    positives, negatives, mask = tensors

    # Exponential of positive and negative logits
    # TODO: -max for numerical stability
    u_p = tf.exp(positives)
    u_ns = tf.exp(negatives)

    # Approximate partition function using negatives
    Z_c = tf.reduce_sum(u_ns, axis=-1)
    if len(Z_c.shape) == 1:
        Z_c = tf.expand_dims(Z_c, axis=-1)

    Z = u_p + Z_c * (vocab_size - 1) / num_negatives

    # Compute Approximate Log Softmax
    log_p = positives - tf.log(Z)
    log_p *= tf.cast(mask, tf.float32)

    # Sum (Multinomial Log Likelihood) over positives
    multi_likelihood = tf.reduce_sum(log_p, axis=-1)

    return -tf.reduce_mean(multi_likelihood)


@dpr.layers.layer(n_in=2, n_out=2)
def AddKL(tensors: Tuple[tf.Tensor, tf.Tensor], beta_start: float, beta_end: float, beta_steps: int):
    """Compute loss + beta * KL, decay beta linearly during training."""
    loss, KL = tensors
    beta = tf.train.polynomial_decay(
        float(beta_start), tf.train.get_global_step(), beta_steps, float(beta_end), power=1.0, cycle=False
    )
    return loss + beta * KL, beta
