# pylint: disable=no-value-for-parameter,invalid-name,unexpected-keyword-arg
"""Multinomial Loss for the Multi-VAE."""

import logging
from typing import Tuple

import tensorflow as tf
import deepr as dpr


LOGGER = logging.getLogger(__name__)


def VAELoss(beta_start: float, beta_end: float, beta_steps: int):
    """Compute Multi-VAE loss."""
    return dpr.layers.Sequential(
        dpr.layers.Select(inputs=("logits", "inputPositivesOneHot", "KL")),
        MultiLoss(inputs=("logits", "inputPositivesOneHot"), outputs="loss_multi"),
        AddKL(
            inputs=("loss_multi", "KL"),
            outputs=("loss", "beta"),
            beta_start=beta_start,
            beta_end=beta_end,
            beta_steps=beta_steps,
        ),
        dpr.layers.Select(inputs=("loss", "loss_multi", "beta")),
    )


@dpr.layers.layer(n_in=2, n_out=1)
def MultiLoss(tensors: Tuple[tf.Tensor]):
    """Multinomial loss."""
    logits, one_hot = tensors
    log_softmax = tf.nn.log_softmax(logits)
    return -tf.reduce_mean(tf.reduce_sum(log_softmax * tf.cast(one_hot, tf.float32), axis=-1))


@dpr.layers.layer(n_in=2, n_out=2)
def AddKL(tensors: Tuple[tf.Tensor], beta_start: float, beta_end: float, beta_steps: int):
    """Compute loss + beta * KL, decay beta linearly during training."""
    loss, KL = tensors
    beta = tf.train.polynomial_decay(
        float(beta_start), tf.train.get_global_step(), beta_steps, float(beta_end), power=1.0, cycle=False
    )
    return loss + beta * KL, beta
