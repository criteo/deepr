# pylint: disable=no-value-for-parameter,invalid-name,unexpected-keyword-arg
"""MultiLogLikelihood Loss with Complementarity Sampling."""

import logging

import deepr


LOGGER = logging.getLogger(__name__)


def MultiLogLikelihoodCSS(vocab_size: int):
    """MultiLogLikelihood Loss with Complementarity Sampling."""
    return deepr.layers.DAG(
        deepr.layers.Select(inputs=("userEmbeddings", "targetPositives", "targetNegatives", "targetMask")),
        deepr.layers.DenseIndex(
            inputs=("userEmbeddings", "targetPositives"),
            outputs="targetPositiveLogits",
            units=vocab_size,
            kernel_name="embeddings",
            bias_name="biases",
            reuse=True,
        ),
        deepr.layers.DenseIndex(
            inputs=("userEmbeddings", "targetNegatives"),
            outputs="targetNegativeLogits",
            units=vocab_size,
            kernel_name="embeddings",
            bias_name="biases",
            reuse=True,
        ),
        deepr.layers.ToFloat(inputs="targetMask", outputs="targetWeight"),
        deepr.layers.ExpandDims(inputs="targetMask", outputs="targetNegativeMask"),
        deepr.layers.MultiLogLikelihoodCSS(
            inputs=("targetPositiveLogits", "targetNegativeLogits", "targetMask", "targetNegativeMask"),
            outputs="loss",
            vocab_size=vocab_size,
        ),
        deepr.layers.TripletPrecision(
            inputs=("targetPositiveLogits", "targetNegativeLogits", "targetNegativeMask", "targetWeight"),
            outputs="triplet_precision",
        ),
        deepr.layers.Select(inputs=("loss", "triplet_precision")),
    )
