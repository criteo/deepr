# pylint: disable=unexpected-keyword-arg,no-value-for-parameter,invalid-name
"""Negative Sampling Loss with biases."""

import deepr as dpr


def NegativeSampling(vocab_size: int):
    """Negative Sampling Loss with biases."""
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
        dpr.layers.ExpandDims(inputs="targetMask", outputs="targetMask"),
        dpr.layers.MaskedNegativeSampling(
            inputs=("targetPositiveLogits", "targetNegativeLogits", "targetMask", "targetWeight"), outputs="loss"
        ),
        dpr.layers.Select(inputs=("loss")),
    )
