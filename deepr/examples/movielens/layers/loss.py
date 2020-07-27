# pylint: disable=unexpected-keyword-arg,no-value-for-parameter
"""BPR Loss."""

import deepr as dpr


def BPRLoss(vocab_size: int, dim: int):
    return dpr.layers.Sequential(
        dpr.layers.Select(inputs=("userEmbeddings", "targetPositives", "targetNegatives", "targetMask")),
        dpr.layers.Embedding(
            inputs="targetPositives",
            outputs="targetPositiveEmbeddings",
            variable_name="embeddings",
            shape=(vocab_size, dim),
            reuse=True,
        ),
        dpr.layers.Embedding(
            inputs="targetNegatives",
            outputs="targetNegativeEmbeddings",
            variable_name="embeddings",
            shape=(vocab_size, dim),
            reuse=True,
        ),
        dpr.layers.DotProduct(inputs=("userEmbeddings", "targetPositiveEmbeddings"), outputs="targetPositiveLogits"),
        dpr.layers.DotProduct(inputs=("userEmbeddings", "targetNegativeEmbeddings"), outputs="targetNegativeLogits"),
        dpr.layers.ToFloat(inputs="targetMask", outputs="targetWeight"),
        dpr.layers.ExpandDims(inputs="targetMask", outputs="targetMask", axis=-1),
        dpr.layers.MaskedBPR(
            inputs=("targetPositiveLogits", "targetNegativeLogits", "targetMask", "targetWeight"), outputs="loss"
        ),
    )
