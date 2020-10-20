# pylint: disable=no-value-for-parameter,invalid-name,unexpected-keyword-arg
"""MultiLogLikelihood Loss with Complementarity Sampling."""

import logging

import deepr as dpr


LOGGER = logging.getLogger(__name__)


def MultiLogLikelihoodCSS(vocab_size: int):
    """MultiLogLikelihood Loss with Complementarity Sampling."""
    return dpr.layers.Sequential(
        dpr.layers.Select(inputs=("userEmbeddings", "targetPositives", "targetNegatives", "targetMask")),
        dpr.layers.DenseIndex(
            inputs=("userEmbeddings", "targetPositives"),
            outputs="targetPositiveLogits",
            units=vocab_size,
            kernel_name="embeddings",
            bias_name="biases",
            reuse=True,
        ),
        dpr.layers.DenseIndex(
            inputs=("userEmbeddings", "targetNegatives"),
            outputs="targetNegativeLogits",
            units=vocab_size,
            kernel_name="embeddings",
            bias_name="biases",
            reuse=True,
        ),
        dpr.layers.ToFloat(inputs="targetMask", outputs="targetWeight"),
        dpr.layers.ExpandDims(inputs="targetMask", outputs="targetNegativeMask"),
        dpr.layers.MultiLogLikelihoodCSS(
            inputs=("targetPositiveLogits", "targetNegativeLogits", "targetMask", "targetNegativeMask"),
            outputs="loss",
            vocab_size=vocab_size,
        ),
        dpr.layers.TripletPrecision(
            inputs=("targetPositiveLogits", "targetNegativeLogits", "targetNegativeMask", "targetWeight"),
            outputs="triplet_precision",
        ),
        dpr.layers.Select(inputs=("loss", "triplet_precision")),
    )
