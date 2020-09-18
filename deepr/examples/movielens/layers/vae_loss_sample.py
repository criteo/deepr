# pylint: disable=no-value-for-parameter,invalid-name,unexpected-keyword-arg
"""Multinomial Loss for the Multi-VAE."""

import logging

import tensorflow as tf
import deepr as dpr


LOGGER = logging.getLogger(__name__)


def VAELossSample(
    beta_start: float, beta_end: float, beta_steps: int, num_negatives: int, vocab_size: int, loss: str = "multi"
):
    """Build Layer for Multi VAE loss with Complementarity Sampling."""
    if loss not in {"bpr", "multi"}:
        raise ValueError(f"Got loss = {loss} (should be either 'bpr' or 'multi')")
    return dpr.layers.Sequential(
        dpr.layers.Select(inputs=("userEmbeddings", "inputPositives", "inputMask", "KL")),
        dpr.layers.ToFloat(inputs="inputMask", outputs="inputWeight"),
        RandomNegatives(
            inputs=("inputPositives", "inputMask"),
            outputs=("inputNegatives", "inputNegativeMask"),
            num_negatives=num_negatives,
            vocab_size=vocab_size,
        ),
        dpr.layers.DenseIndex(
            inputs=("userEmbeddings", "inputPositives"),
            outputs="inputPositiveLogits",
            units=vocab_size,
            kernel_name="embeddings",
            bias_name="biases",
            reuse=True,
        ),
        dpr.layers.DenseIndex(
            inputs=("userEmbeddings", "inputNegatives"),
            outputs="inputNegativeLogits",
            units=vocab_size,
            kernel_name="embeddings",
            bias_name="biases",
            reuse=True,
        ),
        (
            dpr.layers.MultiLogLikelihoodCSS(
                inputs=("inputPositiveLogits", "inputNegativeLogits", "inputMask", "inputNegativeMask"),
                outputs="loss_multi",
                vocab_size=vocab_size,
            )
            if loss == "multi"
            else dpr.layers.MaskedBPR(
                inputs=("inputPositiveLogits", "inputNegativeLogits", "inputNegativeMask", "inputWeight"),
                outputs="loss_bpr",
            )
        ),
        dpr.layers.AddWithWeight(
            inputs=(f"loss_{loss}", "KL"), outputs="loss", start=beta_start, end=beta_end, steps=beta_steps
        ),
        dpr.layers.Select(inputs=("loss", f"loss_{loss}")),
    )


@dpr.layers.layer(n_in=2, n_out=2)
def RandomNegatives(tensors, num_negatives, vocab_size):
    positives, mask = tensors
    negatives = tf.random.uniform(
        shape=[tf.shape(positives)[0], 1, num_negatives], maxval=vocab_size, dtype=tf.int64
    )
    mask = tf.logical_and(tf.ones_like(negatives, dtype=tf.bool), tf.expand_dims(mask, axis=-1))
    return negatives, mask
